#!/usr/bin/env python3
"""Camera-coordinate metric reference: stationary diagnostic only.

Input:   /seg/mask_raw                 mono8, 320x240
Outputs: /lane/camera_reference_debug bgr8
         /lane/metric_reference_debug JSON String

Never publishes /cmd_vel or /cmd_vel_request.
"""

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


W, H = 320, 240
ROAD, WHITE, YELLOW = 1, 2, 3

# BL, BR, TR, TL in the 320x240 camera-coordinate mask.
SRC = np.float32([
    [11.0, 238.0],
    [305.5, 238.0],
    [210.0, 160.0],
    [105.0, 160.0],
])

# Matching measured floor coordinates: x forward, y left, meters.
DST = np.float32([
    [0.305, +0.210],
    [0.305, -0.210],
    [0.930, -0.2115],
    [0.930, +0.2115],
])

# Detect over the visible floor, including positions outside SRC.
# Do not move the top row upward without checking that it is still floor
# and that the homography remains accurate there.
TOP_GROUND_ROW = 140
BOTTOM_GROUND_ROW = 237
ROW_STEP = 3


def camera_to_ground(homography, u, v):
    point = np.float32([[[float(u), float(v)]]])
    return cv2.perspectiveTransform(point, homography)[0, 0].astype(float)


def ground_to_camera(inverse_homography, point):
    value = np.float32([[[float(point[0]), float(point[1])]]])
    uv = cv2.perspectiveTransform(value, inverse_homography)[0, 0]
    return float(uv[0]), float(uv[1])


def candidates_at_row(mask, row):
    """Merge nearby white/yellow fragments on a physical marking."""
    strip = mask[max(0, row - 1):min(H, row + 2)]
    marked = (strip == WHITE) | (strip == YELLOW)
    xs = np.flatnonzero(np.any(marked, axis=0))

    if not len(xs):
        return []

    groups = [[int(xs[0])]]
    for value in xs[1:]:
        value = int(value)
        if value - groups[-1][-1] <= 7:
            groups[-1].append(value)
        else:
            groups.append([value])

    return [
        float(np.median(group))
        for group in groups
        if len(group) >= 2
    ]


def ground_is_usable(mask, u, v):
    """Check that a proposed center is visible on segmented ground."""
    if not np.isfinite([u, v]).all():
        return False

    if not (0 <= u < W and TOP_GROUND_ROW <= v <= BOTTOM_GROUND_ROW):
        return False

    px, py = int(round(u)), int(round(v))
    px = min(W - 1, max(0, px))
    py = min(H - 1, max(0, py))

    patch = mask[
        max(0, py - 2):min(H, py + 3),
        max(0, px - 2):min(W, px + 3),
    ]
    return np.count_nonzero(
        (patch == ROAD) | (patch == WHITE) | (patch == YELLOW)
    ) >= max(1, patch.size // 3)


def closest_match(available, previous_x, max_jump=18.0):
    if previous_x is None or not available:
        return None

    candidate = min(available, key=lambda x: abs(x - previous_x))
    if abs(candidate - previous_x) > max_jump:
        return None

    available.remove(candidate)
    return candidate


class CameraMetricReference(Node):
    def __init__(self):
        super().__init__("camera_metric_reference")
        self.bridge = CvBridge()
        self.homography = cv2.getPerspectiveTransform(SRC, DST)
        self.inverse_homography = np.linalg.inv(self.homography)

        # Only set this if you independently know which boundary
        # is visible at a single-line START. This is a CORRIDOR role,
        # not the line's position within the image.
        self.declare_parameter("single_boundary_side", "unknown")

        # Optional independently measured physical lane width.
        # Zero means learn from a sufficiently good paired observation.
        self.declare_parameter("lane_width_m", 0.0)

        self.learned_width_m = None
        self.last_boundary_traces = {"left": {}, "right": {}}
        self.last_boundary_times = {"left": 0.0, "right": 0.0}

        self.debug_pub = self.create_publisher(
            Image, "/lane/camera_reference_debug", 1
        )
        self.data_pub = self.create_publisher(
            String, "/lane/metric_reference_debug", 1
        )
        self.create_subscription(Image, "/seg/mask_raw", self.on_mask, 1)

        self.get_logger().info(
            "Full visible-ground diagnostic only; no motor commands."
        )

    def publish_result(self, source, debug, result):
        image = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        image.header = source.header
        self.debug_pub.publish(image)

        text = String()
        text.data = json.dumps(result)
        self.data_pub.publish(text)

        log = (
            f"STATUS {result['status']} "
            f"DETAIL {result.get('detail')} "
            f"FAIL_ROW {result.get('failure_row')} "
            f"END {result.get('termination_reason')} "
            f"PAIRS {result['pairs']} SINGLE {result['single']} "
            f"WIDTH {result['width_m']} "
            f"SIDE {result['single_boundary_side']} "
            f"NEAR_X {result['near_x_m']} "
            f"FAR_X {result['far_x_m']} "
            f"SPAN {result.get('observed_span_m')}"
        )
        if result["status"].startswith("INVALID"):
            self.get_logger().warn(log)
        else:
            self.get_logger().info(log)

    def on_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8"
            )
        except Exception as exc:
            self.get_logger().error(f"Mask conversion failed: {exc}")
            return

        if mask.shape != (H, W):
            self.get_logger().error(
                f"Expected 320x240 mask, got {mask.shape[::-1]}; "
                "not applying the homography."
            )
            return

        debug = np.full((H, W, 3), 95, dtype=np.uint8)
        debug[mask == WHITE] = (255, 255, 255)
        debug[mask == YELLOW] = (0, 255, 255)

        # Magenta shows the four points used to FIT the homography.
        # It is no longer a marking-detection boundary.
        polygon = np.rint(SRC).astype(np.int32)
        cv2.polylines(debug, [polygon], True, (255, 0, 255), 1)
        cv2.line(
            debug, (0, TOP_GROUND_ROW), (W - 1, TOP_GROUND_ROW),
            (100, 100, 200), 1
        )

        result = {
            "status": "INVALID_NO_NEAR_ANCHOR",
            "pairs": 0,
            "single": 0,
            "width_m": None,
            "width_source": "UNKNOWN",
            "single_boundary_side": "UNKNOWN",
            "near_x_m": None,
            "far_x_m": None,
            "points_x_forward_y_left_m": [],
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "detail": None,
            "failure_row": None,
            "termination_reason": None,
            "observed_span_m": None,
            "controller_ready": False,
        }

        rows = [
            (row, candidates_at_row(mask, row))
            for row in range(
                BOTTOM_GROUND_ROW, TOP_GROUND_ROW - 1, -ROW_STEP
            )
        ]

        # Find an observation near the car. Do not anchor on an
        # unrelated pair found only far up the image.
        anchor = next(
            (
                (row, candidates)
                for row, candidates in rows
                if row >= 207 and candidates
            ),
            None,
        )
        if anchor is None:
            self.publish_result(msg, debug, result)
            return

        anchor_row, anchor_candidates = anchor
        requested_side = str(
            self.get_parameter("single_boundary_side").value
        ).lower()
        if requested_side not in ("left", "right", "unknown"):
            requested_side = "unknown"

        left_previous = None
        right_previous = None
        anchor_side = None

        now = time.monotonic()
        remembered = {
            side: (
                self.last_boundary_traces[side]
                if now - self.last_boundary_times[side] < 5.0
                else {}
            )
            for side in ("left", "right")
        }

        # Evaluate a CURRENT near-field pair. History helps choose
        # between plausible pairs, but a newly observed pair is not
        # permanently bound to an older curve.
        valid_pairs = []
        for index, left_u in enumerate(anchor_candidates):
            for right_u in anchor_candidates[index + 1:]:
                left_m = camera_to_ground(
                    self.homography, left_u, anchor_row
                )
                right_m = camera_to_ground(
                    self.homography, right_u, anchor_row
                )
                separation = float(
                    np.linalg.norm(left_m - right_m)
                )
                if not 0.20 <= separation <= 0.65:
                    continue

                old_left = remembered["left"].get(anchor_row)
                old_right = remembered["right"].get(anchor_row)
                continues_pair = (
                    old_left is not None
                    and old_right is not None
                    and abs(left_u - old_left) <= 40.0
                    and abs(right_u - old_right) <= 40.0
                )
                straddles_car = left_m[1] > 0.0 > right_m[1]

                if not (continues_pair or straddles_car):
                    continue

                history_error = (
                    abs(left_u - old_left)
                    + abs(right_u - old_right)
                    if continues_pair else float("inf")
                )
                center_error = abs(
                    float(left_m[1] + right_m[1])
                )
                valid_pairs.append(
                    (
                        (0 if continues_pair else 1,
                         history_error, center_error),
                        left_u,
                        right_u,
                    )
                )

        result["role_match_error_px"] = None
        result["role_match_rows"] = 0

        if valid_pairs:
            _, left_previous, right_previous = min(valid_pairs)

        elif len(anchor_candidates) == 1:
            only_u = anchor_candidates[0]

            # Trace the currently visible marking across MANY rows,
            # without yet assigning it a left/right corridor role.
            observed = {}
            previous_u = only_u
            misses = 0

            for row, candidates in rows:
                if row > anchor_row:
                    continue

                if not candidates:
                    misses += 1
                    if misses >= 2:
                        break
                    continue

                candidate = min(
                    candidates,
                    key=lambda u: abs(u - previous_u),
                )
                if abs(candidate - previous_u) > 18.0:
                    break

                observed[row] = candidate
                previous_u = candidate
                misses = 0

            # Compare the SHAPE at matching image rows with EACH
            # remembered physical boundary. A single bottom-row
            # shift can no longer decide the entire association.
            scores = []
            for side in ("left", "right"):
                previous_trace = remembered[side]
                errors = [
                    abs(u - previous_trace[row])
                    for row, u in observed.items()
                    if row in previous_trace
                ]
                if len(errors) >= 4:
                    scores.append(
                        (float(np.median(errors)), side, len(errors))
                    )

            scores.sort()

            if scores and scores[0][0] <= 45.0:
                # If both roles look similarly plausible, do not
                # choose one by color or image-half position.
                if (
                    len(scores) == 1
                    or scores[1][0] - scores[0][0] >= 12.0
                ):
                    error, anchor_side, overlap = scores[0]
                    result["role_match_error_px"] = round(error, 1)
                    result["role_match_rows"] = overlap

            if anchor_side is None:
                # Explicit startup fallback ONLY when no useful
                # history exists. Never override an ambiguous match.
                if not scores:
                    anchor_side = requested_side

            if anchor_side not in ("left", "right"):
                result["status"] = "INVALID_UNKNOWN_BOUNDARY_SIDE"
                self.publish_result(msg, debug, result)
                return

            if anchor_side == "left":
                left_previous = only_u
            else:
                right_previous = only_u

        else:
            result["status"] = "INVALID_AMBIGUOUS_ANCHOR"
            self.publish_result(msg, debug, result)
            return


        # Follow the same physical marking(s) from near to far.
        # A lost track is not replaced by a distant marking.
        left_alive = left_previous is not None
        right_alive = right_previous is not None
        tracked = []  # (camera_row, left_u, right_u)
        missing = 0

        for row, candidates in rows:
            if row > anchor_row:
                continue

            available = list(candidates)
            lu = (
                closest_match(available, left_previous)
                if left_alive else None
            )
            ru = (
                closest_match(available, right_previous)
                if right_alive else None
            )

            if left_alive:
                if lu is None:
                    left_alive = False
                else:
                    left_previous = lu

            if right_alive:
                if ru is None:
                    right_alive = False
                else:
                    right_previous = ru

            if lu is None and ru is None:
                missing += 1
                if missing >= 2:
                    break
                continue

            missing = 0
            tracked.append((row, lu, ru))

            if lu is not None:
                cv2.circle(debug, (round(lu), row), 2,
                           (255, 0, 255), -1)
            if ru is not None:
                cv2.circle(debug, (round(ru), row), 2,
                           (255, 255, 0), -1)

        # Transform paired boundaries first; learn width measured
        # perpendicular to the local direction of the corridor.
        pairs = []
        for row, lu, ru in tracked:
            if lu is None or ru is None or lu >= ru:
                continue

            left_m = camera_to_ground(self.homography, lu, row)
            right_m = camera_to_ground(self.homography, ru, row)
            pairs.append((row, left_m, right_m))

        widths = []
        for index, (_, left_m, right_m) in enumerate(pairs):
            before = 0.5 * (
                pairs[max(0, index - 1)][1]
                + pairs[max(0, index - 1)][2]
            )
            after = 0.5 * (
                pairs[min(len(pairs) - 1, index + 1)][1]
                + pairs[min(len(pairs) - 1, index + 1)][2]
            )
            tangent = after - before
            length = float(np.linalg.norm(tangent))
            if length < 1e-6:
                continue

            left_normal = np.array(
                [-tangent[1], tangent[0]]
            ) / length
            width = float(np.dot(left_m - right_m, left_normal))
            if 0.20 <= width <= 0.65:
                widths.append(width)

        current_width = None
        if len(widths) >= 5:
            median = float(np.median(widths))
            deviation = float(
                np.median(np.abs(np.asarray(widths) - median))
            )
            if deviation <= 0.06:
                current_width = median
                self.learned_width_m = (
                    median if self.learned_width_m is None
                    else 0.8 * self.learned_width_m + 0.2 * median
                )

        configured_width = float(
            self.get_parameter("lane_width_m").value
        )
        if not 0.20 <= configured_width <= 0.65:
            configured_width = None

        if current_width is not None:
            width_used, width_source = current_width, "CURRENT"
        elif configured_width is not None:
            width_used, width_source = configured_width, "PARAMETER"
        elif self.learned_width_m is not None:
            width_used, width_source = self.learned_width_m, "LEARNED"
        else:
            width_used, width_source = None, "UNKNOWN"

        # Compute single-line offsets along a local normal IN METERS.
        # Boundary roles stay fixed when a curve crosses image center.
        singles = []
        for row, lu, ru in tracked:
            if (lu is None) == (ru is None):
                continue
            side = "left" if lu is not None else "right"
            u = lu if lu is not None else ru
            point = camera_to_ground(self.homography, u, row)
            singles.append((row, side, point))

        single_centers = {}
        if width_used is not None and len(singles) >= 2:
            for index, (row, side, point) in enumerate(singles):
                before = singles[max(0, index - 1)][2]
                after = singles[
                    min(len(singles) - 1, index + 1)
                ][2]
                tangent = after - before  # near -> far
                length = float(np.linalg.norm(tangent))
                if length < 1e-6:
                    continue

                left_normal = np.array(
                    [-tangent[1], tangent[0]]
                ) / length
                direction = (
                    -1.0 if side == "left" else +1.0
                )
                single_centers[row] = (
                    point
                    + direction * 0.5 * width_used * left_normal
                )

        paired_centers = {
            row: 0.5 * (left_m + right_m)
            for row, left_m, right_m in pairs
        }

        reference = []
        count_pairs = 0
        count_singles = 0
        previous_row = None
        previous_center = None
        skipped_nonprogress = 0
        failure = None
        failure_row = None
        termination = "END_OF_TRACKED_OBSERVATIONS"

        for row, _, _ in tracked:
            if row in paired_centers:
                center = paired_centers[row]
                mode = "PAIR"
            elif row in single_centers:
                center = single_centers[row]
                mode = "SINGLE"
            else:
                if reference:
                    failure = "MISSING_CENTER_IN_TRACK"
                    failure_row = row
                    break
                continue

            if not np.isfinite(center).all():
                failure = "NONFINITE_METRIC_CENTER"
                failure_row = row
                break

            u, v = ground_to_camera(
                self.inverse_homography, center
            )

            if not np.isfinite([u, v]).all():
                failure = "NONFINITE_CAMERA_PROJECTION"
                failure_row = row
                break

            # A center leaving the camera's visible floor ends the
            # observation. It does not erase the connected prefix.
            if not (
                0 <= u < W
                and TOP_GROUND_ROW <= v <= BOTTOM_GROUND_ROW
            ):
                termination = "CENTER_LEFT_VISIBLE_IMAGE"
                failure_row = row
                break

            # A center projected inside the image but classified as
            # non-ground is different: do not call it a valid lane.
            if not ground_is_usable(mask, u, v):
                failure = "CENTER_NOT_SEGMENTED_GROUND"
                failure_row = row
                break

            if previous_center is not None:
                row_gap = previous_row - row
                center_jump = float(
                    np.linalg.norm(center - previous_center)
                )
                forward_step = float(
                    center[0] - previous_center[0]
                )

                if row_gap > 9:
                    failure = (
                        f"ROW_GAP_{row_gap}_PX"
                    )
                    failure_row = row
                    break

                if center_jump > 0.13:
                    failure = (
                        f"CENTER_JUMP_{center_jump:.3f}_M"
                    )
                    failure_row = row
                    break

                # Do not fail on a sub-millimeter forward reversal
                # from segmentation/homography rounding. Omit that
                # point so published x values remain increasing.
                if forward_step <= 0.002:
                    skipped_nonprogress += 1
                    if skipped_nonprogress > 2:
                        failure = (
                            f"REPEATED_NONPROGRESS_"
                            f"{forward_step:.4f}_M"
                        )
                        failure_row = row
                        break
                    continue

            skipped_nonprogress = 0
            reference.append(
                (float(center[0]), float(center[1]))
            )
            previous_row = row
            previous_center = center

            if mode == "PAIR":
                count_pairs += 1
                color = (255, 0, 0)    # Blue: paired center
            else:
                count_singles += 1
                color = (0, 140, 255)  # Orange: single center

            cv2.circle(
                debug, (round(u), round(v)), 2, color, -1
            )

        result["pairs"] = count_pairs
        result["single"] = count_singles
        result["width_m"] = (
            round(width_used, 4)
            if width_used is not None else None
        )
        result["width_source"] = width_source
        result["single_boundary_side"] = (
            anchor_side.upper()
            if anchor_side is not None else "TRACKED_FROM_PAIR"
        )
        result["failure_row"] = failure_row
        result["termination_reason"] = termination

        if reference:
            near_x = reference[0][0]
            far_x = reference[-1][0]
            span:float = far_x - near_x

            result["near_x_m"] = round(near_x, 4)
            result["far_x_m"] = round(far_x, 4)
            result["observed_span_m"] = round(span, 4)
        else:
            near_x = far_x = span = None

        if failure is not None:
            result["status"] = "INVALID_REFERENCE_GEOMETRY"
            result["detail"] = failure

        elif not reference:
            if singles and width_used is None and not pairs:
                result["status"] = "INVALID_WIDTH_UNKNOWN"
                result["detail"] = (
                    "One boundary tracked, but no measured "
                    "or learned lane width"
                )
            else:
                result["status"] = "INVALID_NO_METRIC_REFERENCE"
                result["detail"] = (
                    "No center passed the visible-ground checks"
                )

        elif len(reference) < 6 or span < 0.08:
            result["status"] = "INVALID_INSUFFICIENT_SPAN"
            result["detail"] = (
                f"{len(reference)} points; "
                f"{span:.3f} m forward span"
            )

        elif near_x > 0.45:
            result["status"] = "INVALID_NO_NEAR_COVERAGE"
            result["detail"] = (
                f"First observed center is {near_x:.3f} m "
                "ahead; near corridor is unobserved"
            )

        elif far_x >= 0.70 and span >= 0.25:
            result["status"] = "VALID_FOR_DIAGNOSTIC_TEST"
            result["detail"] = (
                "Connected observed reference with full "
                "diagnostic coverage"
            )

        else:
            result["status"] = (
                "USABLE_SHORT_REFERENCE_DIAGNOSTIC"
            )
            result["detail"] = (
                f"Connected observed path ends at "
                f"{far_x:.3f} m; below full-coverage requirement"
            )

        # Publish observed points for both diagnostic categories.
        # Genuine invalid results publish NO metric path.
        if result["status"] in (
            "VALID_FOR_DIAGNOSTIC_TEST",
            "USABLE_SHORT_REFERENCE_DIAGNOSTIC",
        ):
            result["points_x_forward_y_left_m"] = [
                [round(x, 4), round(y, 4)]
                for x, y in reference
            ]

            # Refresh boundary memory only from an accepted,
            # connected observation. An invalid path cannot teach
            # the next frame which corridor to follow.
            if len(tracked) >= 4:
                for side, index in (
                    ("left", 1),
                    ("right", 2),
                ):
                    trace = {
                        row: values[index]
                        for values in tracked
                        for row in [values[0]]
                        if values[index] is not None
                    }
                    if len(trace) >= 4:
                        self.last_boundary_traces[side] = trace
                        self.last_boundary_times[side] = (
                            time.monotonic()
                        )

        self.publish_result(msg, debug, result)


def main(args=None):
    rclpy.init(args=args)
    node = CameraMetricReference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()