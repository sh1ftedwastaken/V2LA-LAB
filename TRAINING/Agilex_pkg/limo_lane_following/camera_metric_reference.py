#!/usr/bin/env python3
"""Corridor-associated camera metric reference. Diagnostic only.

Input:
    /seg/mask_raw                  sensor_msgs/Image, mono8, 320x240

Outputs:
    /lane/camera_reference_debug   sensor_msgs/Image, bgr8
    /lane/metric_reference_debug   std_msgs/String, JSON

No /cmd_vel or /cmd_vel_request publisher.

Conventions: X forward, Y left, relative to the front bumper.
Colors describe segmentation classes, NOT fixed corridor roles.
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
TOP_ROW, BOTTOM_ROW, ROW_STEP = 140, 237, 3

SRC = np.float32([
    [11.0,  238.0],
    [305.5, 238.0],
    [210.0, 160.0],
    [105.0, 160.0],
])
DST = np.float32([
    [0.305, +0.210],
    [0.305, -0.210],
    [0.930, -0.2115],
    [0.930, +0.2115],
])

FULL_STATUS = "VALID_FOR_DIAGNOSTIC_TEST"
SHORT_STATUS = "USABLE_SHORT_REFERENCE_DIAGNOSTIC"


def to_ground(homography, u, row):
    uv = np.float32([[[float(u), float(row)]]])
    return cv2.perspectiveTransform(uv, homography)[0, 0].astype(float)


def to_camera(inverse_homography, point):
    xy = np.float32([[[float(point[0]), float(point[1])]]])
    uv = cv2.perspectiveTransform(xy, inverse_homography)[0, 0]
    return float(uv[0]), float(uv[1])


def ground_patch_ok(mask, u, v):
    if not np.isfinite([u, v]).all():
        return False

    if not (0 <= u < W and TOP_ROW <= v <= BOTTOM_ROW):
        return False

    px = int(np.clip(round(u), 0, W - 1))
    py = int(np.clip(round(v), 0, H - 1))
    patch = mask[
        max(0, py - 2):min(H, py + 3),
        max(0, px - 2):min(W, px + 3),
    ]
    ground = (
        (patch == ROAD) | (patch == WHITE) | (patch == YELLOW)
    )
    return np.count_nonzero(ground) >= max(1, patch.size // 3)


def vector_normal(points, index):
    """Left-pointing unit normal to a locally observed marking."""
    lo = max(0, index - 2)
    hi = min(len(points) - 1, index + 2)

    tangent = points[hi] - points[lo]
    length = float(np.linalg.norm(tangent))
    if length < 1e-5:
        return None

    tangent /= length

    # Rows are stored from near to far. A valid trace must have a
    # generally forward-directed local tangent.
    if tangent[0] <= 0:
        return None

    return np.array([-tangent[1], tangent[0]], dtype=float)


def extract_traces(mask, homography):
    """Extract separate connected marking traces, retaining color.

    Connected components prevent the median of two same-color
    markings from silently becoming a third, nonexistent marking.
    """
    traces = []
    next_id = 0

    for color_id, name in ((WHITE, "WHITE"), (YELLOW, "YELLOW")):
        roi = (mask[TOP_ROW:BOTTOM_ROW + 1] == color_id).astype(
            np.uint8
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            roi, connectivity=8
        )

        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 12:
                continue

            samples = []
            for row in range(BOTTOM_ROW, TOP_ROW - 1, -ROW_STEP):
                local_row = row - TOP_ROW
                xs = np.flatnonzero(labels[local_row] == component)
                if len(xs) < 2:
                    continue

                u = float(np.median(xs))
                xy = to_ground(homography, u, row)
                if np.isfinite(xy).all():
                    samples.append({
                        "row": row,
                        "u": u,
                        "xy": xy,
                    })

            if len(samples) < 4:
                continue

            traces.append({
                "id": next_id,
                "color": name,
                "area": area,
                "samples": samples,
                "by_row": {s["row"]: s for s in samples},
            })
            next_id += 1

    return traces


def trace_points(trace):
    return np.asarray(
        [sample["xy"] for sample in trace["samples"]],
        dtype=float,
    )


def trace_normals(trace):
    points = trace_points(trace)
    return [
        vector_normal(points, index)
        for index in range(len(points))
    ]


def overlap_error(first, second):
    """Median metric displacement at the same sampled image rows."""
    common = set(first["by_row"]).intersection(second["by_row"])
    if len(common) < 4:
        return None

    errors = [
        float(np.linalg.norm(
            first["by_row"][row]["xy"]
            - second["by_row"][row]["xy"]
        ))
        for row in common
    ]
    return float(np.median(errors))


def path_arc(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(
        np.diff(points, axis=0), axis=1
    )))


class CameraMetricReference(Node):
    def __init__(self):
        super().__init__("camera_metric_reference")
        self.bridge = CvBridge()
        self.homography = cv2.getPerspectiveTransform(SRC, DST)
        self.inverse_homography = np.linalg.inv(self.homography)

        # This is a measured/default width, not evidence of which
        # side of a solitary marking contains the intended corridor.
        self.declare_parameter("lane_width_m", 0.39)

        # Only for a cold start when a SINGLE boundary is visible
        # and its corridor role is independently known.
        self.declare_parameter("single_boundary_side", "unknown")

        self.learned_width_m = None
        self.role_colors = {"left": None, "right": None}
        self.previous_traces = {"left": None, "right": None}
        self.previous_time = None
        self.corridor_established = False

        self.debug_pub = self.create_publisher(
            Image, "/lane/camera_reference_debug", 10
        )
        self.data_pub = self.create_publisher(
            String, "/lane/metric_reference_debug", 10
        )
        self.create_subscription(
            Image, "/seg/mask_raw", self.on_mask, 10
        )

        self.get_logger().info(
            "Corridor-associated reference diagnostic active; "
            "no motion commands."
        )

    def blank_result(self, msg):
        return {
            "status": "INVALID_NO_ASSOCIATED_BOUNDARY",
            "detail": None,
            "termination_reason": None,
            "anchor_mode": None,
            "pair_samples": 0,
            "single_samples": 0,
            "pairs": 0,
            "single": 0,
            "single_boundary_side": "UNKNOWN",
            "width_m": None,
            "width_source": "UNKNOWN",
            "near_x_m": None,
            "far_x_m": None,
            "max_x_m": None,
            "observed_span_m": None,
            "observed_arc_m": None,
            "points_x_forward_y_left_m": [],
            "corridor_established": self.corridor_established,
            "controller_ready": False,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
        }

    def publish_result(self, msg, debug, result):
        image = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        image.header = msg.header
        self.debug_pub.publish(image)

        text = String()
        text.data = json.dumps(result)
        self.data_pub.publish(text)

        line = (
            f"STATUS {result['status']} "
            f"DETAIL {result['detail']} "
            f"MODE {result['anchor_mode']} "
            f"END {result['termination_reason']} "
            f"PAIR/SINGLE {result['pairs']}/{result['single']} "
            f"ROLE {result['single_boundary_side']} "
            f"WIDTH {result['width_m']}({result['width_source']}) "
            f"NEAR/MAX_X {result['near_x_m']}/{result['max_x_m']} "
            f"ARC {result['observed_arc_m']}"
        )

        if result["status"].startswith("INVALID"):
            self.get_logger().warn(line)
        else:
            self.get_logger().info(line)

    def remembered_error(self, trace, side, now):
        previous = self.previous_traces[side]
        if previous is None or self.previous_time is None:
            return None

        # Compare observations only while successive camera frames
        # remain reasonably close in time. After a longer gap, color
        # identity can still help, but geometry is no longer predicted.
        if now - self.previous_time > 0.75:
            return None

        return overlap_error(trace, previous)

    def pair_candidates(self, traces, now):
        proposals = []

        for left in traces:
            for right in traces:
                if left["id"] == right["id"]:
                    continue

                common = sorted(
                    set(left["by_row"]).intersection(
                        right["by_row"]
                    ),
                    reverse=True,
                )
                if len(common) < 4:
                    continue

                # Both boundaries must maintain left/right ordering
                # throughout their common observations.
                lateral = np.array([
                    left["by_row"][row]["xy"][1]
                    - right["by_row"][row]["xy"][1]
                    for row in common
                ])
                if np.any(lateral <= 0.08):
                    continue

                left_normals = {
                    sample["row"]: normal
                    for sample, normal in zip(
                        left["samples"], trace_normals(left)
                    )
                }
                right_normals = {
                    sample["row"]: normal
                    for sample, normal in zip(
                        right["samples"], trace_normals(right)
                    )
                }

                widths = []
                for row in common:
                    ln = left_normals.get(row)
                    rn = right_normals.get(row)
                    if ln is None or rn is None:
                        continue

                    normal = ln + rn
                    length = float(np.linalg.norm(normal))
                    if length < 1e-5:
                        continue
                    normal /= length

                    separation = (
                        left["by_row"][row]["xy"]
                        - right["by_row"][row]["xy"]
                    )
                    widths.append(float(np.dot(
                        separation, normal
                    )))

                if len(widths) < 4:
                    continue

                width = float(np.median(widths))
                deviation = float(np.median(
                    np.abs(np.asarray(widths) - width)
                ))
                if not (0.27 <= width <= 0.52):
                    continue
                if deviation > 0.06:
                    continue

                first_row = common[0]
                near_center = 0.5 * (
                    left["by_row"][first_row]["xy"]
                    + right["by_row"][first_row]["xy"]
                )

                if near_center[0] > 0.65:
                    continue

                # If roles were learned from an established pair,
                # do not reverse their color identities.
                if self.role_colors["left"] is not None:
                    if left["color"] != self.role_colors["left"]:
                        continue
                if self.role_colors["right"] is not None:
                    if right["color"] != self.role_colors["right"]:
                        continue

                left_error = self.remembered_error(
                    left, "left", now
                )
                right_error = self.remembered_error(
                    right, "right", now
                )

                recent_match = (
                    left_error is not None
                    and left_error <= 0.20
                ) or (
                    right_error is not None
                    and right_error <= 0.20
                )

                # For a new corridor, require the pair to pass
                # near the car. After an established corridor,
                # either near-field evidence or recent boundary
                # continuity can support a farther-overlap pair.
                if not self.corridor_established:
                    if near_center[0] > 0.45:
                        continue
                    if abs(float(near_center[1])) > 0.22:
                        continue
                elif not recent_match:
                    if (
                        near_center[0] > 0.45
                        or abs(float(near_center[1])) > 0.20
                    ):
                        continue

                width_target = (
                    self.learned_width_m
                    if self.learned_width_m is not None
                    else 0.39
                )
                score = (
                    0.0 if recent_match else 0.10
                ) + (
                    min(left_error, 0.30)
                    if left_error is not None else 0.06
                ) + (
                    min(right_error, 0.30)
                    if right_error is not None else 0.06
                ) + (
                    abs(width - width_target) * 0.4
                ) + (
                    abs(float(near_center[1])) * 0.2
                ) - (
                    min(len(common), 25) * 0.002
                )

                proposals.append({
                    "score": score,
                    "left": left,
                    "right": right,
                    "width": width,
                    "shared_rows": common,
                    "recent_match": recent_match,
                })

        return sorted(proposals, key=lambda item: item["score"])

    def single_candidates(self, traces, now, requested_side):
        proposals = []

        for trace in traces:
            for side in ("left", "right"):
                known_color = self.role_colors[side]
                other_color = self.role_colors[
                    "right" if side == "left" else "left"
                ]
                match_error = self.remembered_error(
                    trace, side, now
                )

                if known_color is not None:
                    if trace["color"] != known_color:
                        continue
                    if (
                        match_error is not None
                        and match_error > 0.20
                    ):
                        continue
                    role_source = "ESTABLISHED_CORRIDOR"
                elif (
                    not self.corridor_established
                    and requested_side == side
                    and len(traces) == 1
                ):
                    role_source = "EXPLICIT_SINGLE_START"
                else:
                    continue

                # If color is shared across both roles, color alone
                # cannot maintain identity after geometry is lost.
                if (
                    other_color == trace["color"]
                    and match_error is None
                ):
                    continue

                near_x = float(trace["samples"][0]["xy"][0])
                if near_x > 0.65:
                    continue

                score = (
                    match_error if match_error is not None
                    else 0.09
                ) + (
                    max(0.0, near_x - 0.40)
                ) - (
                    min(len(trace["samples"]), 30) * 0.001
                )

                proposals.append({
                    "score": score,
                    "trace": trace,
                    "side": side,
                    "role_source": role_source,
                })

        return sorted(proposals, key=lambda item: item["score"])

    def width_for_frame(self, pair):
        configured = float(
            self.get_parameter("lane_width_m").value
        )
        if not 0.27 <= configured <= 0.52:
            configured = None

        if pair is not None:
            return pair["width"], "CURRENT"
        if self.learned_width_m is not None:
            return self.learned_width_m, "LEARNED"
        if configured is not None:
            return configured, "PARAMETER"
        return None, "UNKNOWN"

    def make_single_centers(self, trace, side, width):
        if width is None:
            return {}

        samples = trace["samples"]
        normals = trace_normals(trace)
        direction = -1.0 if side == "left" else +1.0

        centers = {}
        for sample, normal in zip(samples, normals):
            if normal is None:
                continue
            centers[sample["row"]] = (
                sample["xy"]
                + direction * 0.5 * width * normal
            )
        return centers

    def make_pair_centers(self, pair):
        left = pair["left"]["by_row"]
        right = pair["right"]["by_row"]

        return {
            row: 0.5 * (
                left[row]["xy"] + right[row]["xy"]
            )
            for row in pair["shared_rows"]
        }

    def validate_path(self, mask, centers, debug, result):
        """Keep a connected, observed prefix; never bridge a jump."""
        accepted = []
        modes = []
        last_row = None
        last_xy = None
        termination = "END_OF_OBSERVED_TRACE"

        for row in sorted(centers, reverse=True):
            xy, mode = centers[row]

            if not np.isfinite(xy).all():
                termination = f"NONFINITE_CENTER_ROW_{row}"
                break

            u, v = to_camera(self.inverse_homography, xy)
            if not np.isfinite([u, v]).all():
                termination = f"NONFINITE_PROJECTION_ROW_{row}"
                break

            if not (
                0 <= u < W and TOP_ROW <= v <= BOTTOM_ROW
            ):
                if mode == "SINGLE":
                    # The center is inferred from a valid boundary, 
                    # allow it to project outside the visible image.
                    pass
                else:
                    if not accepted:
                        continue
                    termination = f"CENTER_LEFT_IMAGE_ROW_{row}"
                    break

            elif not ground_patch_ok(mask, u, v):
                if mode == "SINGLE":
                    # The center is inferred, allow it to project 
                    # onto unsegmented pixels.
                    pass
                else:
                    if not accepted:
                        continue
                    termination = f"CENTER_NOT_GROUND_ROW_{row}"
                    break

            if last_xy is not None:
                row_gap = last_row - row
                distance = float(np.linalg.norm(xy - last_xy))

                if row_gap > 9:
                    termination = f"ROW_GAP_{row_gap}_PX_AT_{row}"
                    break

                # Far-image quantization grows in metric units.
                # Do not accept the jump; evaluate the prefix.
                if distance > max(
                    0.15, 0.20 * max(xy[0], last_xy[0])
                ):
                    termination = (
                        f"CENTER_JUMP_{distance:.3f}_M_AT_{row}"
                    )
                    break

            accepted.append((row, xy))
            modes.append(mode)
            last_row = row
            last_xy = xy

            color = (
                (255, 0, 0) if mode == "PAIR"
                else (0, 140, 255)
            )
            cv2.circle(
                debug, (round(u), round(v)), 2, color, -1
            )

        result["termination_reason"] = termination
        result["pairs"] = modes.count("PAIR")
        result["single"] = modes.count("SINGLE")
        result["pair_samples"] = result["pairs"]
        result["single_samples"] = result["single"]

        if not accepted:
            result["status"] = "INVALID_NO_METRIC_REFERENCE"
            result["detail"] = (
                "No associated center was observed on visible "
                "segmented ground"
            )
            return False

        points = np.asarray(
            [xy for _, xy in accepted], dtype=float
        )
        near_x = float(points[0, 0])
        far_x = float(points[-1, 0])
        max_x = float(np.max(points[:, 0]))
        arc = path_arc(points)

        result["near_x_m"] = round(near_x, 4)
        result["far_x_m"] = round(far_x, 4)
        result["max_x_m"] = round(max_x, 4)
        result["observed_span_m"] = round(
            max_x - near_x, 4
        )
        result["observed_arc_m"] = round(arc, 4)

        if len(points) < 5 or arc < 0.14:
            result["status"] = "INVALID_INSUFFICIENT_PATH"
            result["detail"] = (
                f"{len(points)} points, {arc:.3f} m arc"
            )
            return False

        if near_x > 0.60:
            result["status"] = "INVALID_NO_NEAR_COVERAGE"
            result["detail"] = (
                f"First visible center is {near_x:.3f} m ahead"
            )
            return False

        forward_steps = np.diff(points[:, 0])
        strictly_forward = bool(
            np.all(forward_steps > 0.0001)
        )

        if (
            near_x <= 0.45
            and max_x >= 0.70
            and arc >= 0.30
            and strictly_forward
        ):
            result["status"] = FULL_STATUS
            result["detail"] = (
                "Associated corridor; full observed coverage"
            )
        else:
            result["status"] = SHORT_STATUS
            result["detail"] = (
                "Associated corridor; limited or nonmonotonic "
                "forward coverage"
            )

        # These are observed diagnostic points. A short curved path
        # need not be representable as y(x); the downstream filter
        # must NOT assume otherwise.
        result["points_x_forward_y_left_m"] = [
            [round(float(x), 4), round(float(y), 4)]
            for x, y in points
        ]
        result["strictly_increasing_x"] = strictly_forward
        return True

    def on_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8"
            )
        except Exception as exc:
            self.get_logger().error(
                f"Mask conversion failed: {exc}"
            )
            return

        if mask.shape != (H, W):
            self.get_logger().error(
                f"Expected 320x240 mask; received {mask.shape}"
            )
            return

        debug = np.full((H, W, 3), 95, dtype=np.uint8)
        debug[mask == WHITE] = (255, 255, 255)
        debug[mask == YELLOW] = (0, 255, 255)
        cv2.polylines(
            debug, [np.rint(SRC).astype(np.int32)],
            True, (255, 0, 255), 1
        )
        cv2.line(
            debug, (0, TOP_ROW), (W - 1, TOP_ROW),
            (100, 100, 200), 1
        )

        result = self.blank_result(msg)
        traces = extract_traces(mask, self.homography)
        result["trace_count"] = len(traces)
        result["white_traces"] = sum(
            t["color"] == "WHITE" for t in traces
        )
        result["yellow_traces"] = sum(
            t["color"] == "YELLOW" for t in traces
        )

        now = time.monotonic()
        requested_side = str(
            self.get_parameter("single_boundary_side").value
        ).lower()
        if requested_side not in (
            "left", "right", "unknown"
        ):
            requested_side = "unknown"

        pairs = self.pair_candidates(traces, now)
        singles = self.single_candidates(
            traces, now, requested_side
        )

        # Two nearly equal pair hypotheses are not a reliable
        # corridor choice.
        pair = None
        if pairs:
            if (
                len(pairs) == 1
                or pairs[1]["score"] - pairs[0]["score"] >= 0.035
            ):
                pair = pairs[0]

        if pair is None and len(pairs) >= 2:
            result["status"] = "INVALID_CORRIDOR_AMBIGUOUS"
            result["detail"] = (
                "Competing paired corridor hypotheses"
            )
            self.publish_result(msg, debug, result)
            return

        single = None
        if singles:
            if (
                len(singles) == 1
                or singles[1]["score"] - singles[0]["score"]
                >= 0.035
            ):
                single = singles[0]

        if pair is None and single is None:
            result["detail"] = (
                "No pair or role-known single trace could be "
                "associated with the selected corridor"
            )
            self.publish_result(msg, debug, result)
            return

        width, width_source = self.width_for_frame(pair)
        result["width_m"] = (
            round(width, 4) if width is not None else None
        )
        result["width_source"] = width_source

        pair_centers = (
            self.make_pair_centers(pair)
            if pair is not None else {}
        )

        # A single-line hypothesis works across its ENTIRE trace.
        # It does not require two rows labelled "single-only".
        single_centers = {}
        single_trace = None
        single_side = None

        if pair is not None:
            # Prefer the boundary with more observed rows when
            # extending beyond the paired overlap.
            extensions = []
            for side in ("left", "right"):
                trace = pair[side]
                candidate = self.make_single_centers(
                    trace, side, width
                )
                extensions.append(
                    (len(candidate), side, trace, candidate)
                )

            extensions.sort(
                key=lambda item: item[0], reverse=True
            )
            _, single_side, single_trace, single_centers = (
                extensions[0]
            )
            result["anchor_mode"] = "PAIR_WITH_SINGLE_EXTENSION"

        else:
            single_trace = single["trace"]
            single_side = single["side"]
            single_centers = self.make_single_centers(
                single_trace, single_side, width
            )
            result["anchor_mode"] = single["role_source"]

        result["single_boundary_side"] = single_side.upper()

        if width is None and not pair_centers:
            result["status"] = "INVALID_WIDTH_UNKNOWN"
            result["detail"] = (
                "Single boundary has no measured or configured width"
            )
            self.publish_result(msg, debug, result)
            return

        centers = {}
        for row, xy in single_centers.items():
            centers[row] = (xy, "SINGLE")

        for row, xy in pair_centers.items():
            centers[row] = (xy, "PAIR")

        # Check whether using the lone boundary would introduce a
        # jump at the paired-to-single transition. Do not join paths
        # belonging to different corridors.
        shared = set(pair_centers).intersection(single_centers)
        if shared:
            disagreement = float(np.median([
                np.linalg.norm(
                    pair_centers[row] - single_centers[row]
                )
                for row in shared
            ]))
            result["pair_single_disagreement_m"] = round(
                disagreement, 4
            )
            if disagreement > 0.12:
                result["status"] = "INVALID_PAIR_SINGLE_CONFLICT"
                result["detail"] = (
                    f"Pair and surviving boundary disagree by "
                    f"{disagreement:.3f} m"
                )
                self.publish_result(msg, debug, result)
                return

        accepted = self.validate_path(
            mask, centers, debug, result
        )

        if accepted:
            # Only a verified pair establishes/re-establishes both
            # boundary roles and teaches the current width.
            if pair is not None:
                if pair["left"]["color"] != pair["right"]["color"]:
                    self.role_colors = {
                        "left": pair["left"]["color"],
                        "right": pair["right"]["color"],
                    }

                self.corridor_established = True
                self.learned_width_m = (
                    pair["width"]
                    if self.learned_width_m is None
                    else 0.85 * self.learned_width_m
                    + 0.15 * pair["width"]
                )
                self.previous_traces["left"] = pair["left"]
                self.previous_traces["right"] = pair["right"]

            elif single is not None:
                self.previous_traces[
                    single["side"]
                ] = single["trace"]

            self.previous_time = now

        result["corridor_established"] = (
            self.corridor_established
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