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

        self.declare_parameter("single_boundary_side", "unknown")
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
            f"END {result.get('termination_reason')} "
            f"PAIRS {result['pairs']} SINGLE {result['single']} "
            f"SIDE {result['single_boundary_side']} "
            f"SPAN {result.get('observed_span_m')}"
        )
        if result["status"].startswith("INVALID"):
            self.get_logger().warn(log)
        else:
            self.get_logger().info(log)

    def on_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            return

        debug = np.full((H, W, 3), 95, dtype=np.uint8)
        debug[mask == WHITE] = (255, 255, 255)
        debug[mask == YELLOW] = (0, 255, 255)

        polygon = np.rint(SRC).astype(np.int32)
        cv2.polylines(debug, [polygon], True, (255, 0, 255), 1)
        cv2.line(debug, (0, TOP_GROUND_ROW), (W - 1, TOP_GROUND_ROW), (100, 100, 200), 1)

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
        }

        rows = [
            (row, candidates_at_row(mask, row))
            for row in range(BOTTOM_GROUND_ROW, TOP_GROUND_ROW - 1, -ROW_STEP)
        ]

        requested_side = str(self.get_parameter("single_boundary_side").value).lower()
        if requested_side not in ("left", "right", "unknown"):
            requested_side = "unknown"

        now = time.monotonic()
        remembered = {
            side: (self.last_boundary_traces[side] if now - self.last_boundary_times[side] < 5.0 else {})
            for side in ("left", "right")
        }

        anchor_row = None
        left_previous = None
        right_previous = None
        anchor_side = None

        # Scan a BAND of near rows for the first trustworthy anchor (row >= 207)
        for row, candidates in rows:
            if row < 160:
                continue

            valid_pairs = []
            for index, left_u in enumerate(candidates):
                for right_u in candidates[index + 1:]:
                    left_m = camera_to_ground(self.homography, left_u, row)
                    right_m = camera_to_ground(self.homography, right_u, row)
                    separation = float(np.linalg.norm(left_m - right_m))
                    if not 0.20 <= separation <= 0.65:
                        continue

                    old_left = remembered["left"].get(row)
                    old_right = remembered["right"].get(row)
                    continues_pair = (old_left is not None and old_right is not None and 
                                      abs(left_u - old_left) <= 40.0 and abs(right_u - old_right) <= 40.0)
                    straddles_car = left_m[1] > 0.0 > right_m[1]

                    if continues_pair or straddles_car:
                        history_error = (abs(left_u - old_left) + abs(right_u - old_right)) if continues_pair else float("inf")
                        center_error = abs(float(left_m[1] + right_m[1]))
                        valid_pairs.append(((0 if continues_pair else 1, history_error, center_error), left_u, right_u))

            if valid_pairs:
                _, left_previous, right_previous = min(valid_pairs)
                anchor_row = row
                break

            elif len(candidates) == 1:
                only_u = candidates[0]
                scores = []
                for side in ("left", "right"):
                    prev_trace = remembered[side]
                    if row in prev_trace and abs(only_u - prev_trace[row]) <= 40.0:
                        scores.append((abs(only_u - prev_trace[row]), side))

                scores.sort()
                
                if scores:
                    anchor_side = scores[0][1]
                elif requested_side != "unknown":
                    anchor_side = requested_side
                
                if anchor_side == "left":
                    left_previous = only_u
                    anchor_row = row
                    break
                elif anchor_side == "right":
                    right_previous = only_u
                    anchor_row = row
                    break

        if anchor_row is None:
            self.publish_result(msg, debug, result)
            return

        left_alive = left_previous is not None
        right_alive = right_previous is not None
        tracked = [] 
        missing = 0

        for row, candidates in rows:
            if row > anchor_row:
                continue

            available = list(candidates)
            lu = closest_match(available, left_previous) if left_alive else None
            ru = closest_match(available, right_previous) if right_alive else None

            if left_alive:
                if lu is None: left_alive = False
                else: left_previous = lu

            if right_alive:
                if ru is None: right_alive = False
                else: right_previous = ru

            if lu is None and ru is None:
                missing += 1
                if missing >= 2: break
                continue

            missing = 0
            tracked.append((row, lu, ru))

            if lu is not None: cv2.circle(debug, (round(lu), row), 2, (255, 0, 255), -1)
            if ru is not None: cv2.circle(debug, (round(ru), row), 2, (255, 255, 0), -1)

        pairs = []
        for row, lu, ru in tracked:
            if lu is None or ru is None or lu >= ru: continue
            left_m = camera_to_ground(self.homography, lu, row)
            right_m = camera_to_ground(self.homography, ru, row)
            pairs.append((row, left_m, right_m))

        widths = []
        for index, (_, left_m, right_m) in enumerate(pairs):
            before = 0.5 * (pairs[max(0, index - 1)][1] + pairs[max(0, index - 1)][2])
            after = 0.5 * (pairs[min(len(pairs) - 1, index + 1)][1] + pairs[min(len(pairs) - 1, index + 1)][2])
            tangent = after - before
            length = float(np.linalg.norm(tangent))
            if length >= 1e-6:
                left_normal = np.array([-tangent[1], tangent[0]]) / length
                width = float(np.dot(left_m - right_m, left_normal))
                if 0.20 <= width <= 0.65: widths.append(width)

        current_width = None
        if len(widths) >= 5:
            median = float(np.median(widths))
            if float(np.median(np.abs(np.asarray(widths) - median))) <= 0.06:
                current_width = median
                self.learned_width_m = median if self.learned_width_m is None else 0.8 * self.learned_width_m + 0.2 * median

        configured_width = float(self.get_parameter("lane_width_m").value)
        if not 0.20 <= configured_width <= 0.65: configured_width = None

        width_used = current_width if current_width is not None else configured_width if configured_width is not None else self.learned_width_m
        width_source = "CURRENT" if current_width is not None else "PARAMETER" if configured_width is not None else "LEARNED" if self.learned_width_m is not None else "UNKNOWN"

        singles = []
        for row, lu, ru in tracked:
            if (lu is None) == (ru is None): continue
            side = "left" if lu is not None else "right"
            u = lu if lu is not None else ru
            point = camera_to_ground(self.homography, u, row)
            singles.append((row, side, point))

        single_centers = {}
        if width_used is not None and len(singles) >= 2:
            for index, (row, side, point) in enumerate(singles):
                before = singles[max(0, index - 1)][2]
                after = singles[min(len(singles) - 1, index + 1)][2]
                tangent = after - before 
                length = float(np.linalg.norm(tangent))
                if length >= 1e-6:
                    left_normal = np.array([-tangent[1], tangent[0]]) / length
                    direction = -1.0 if side == "left" else +1.0
                    single_centers[row] = point + direction * 0.5 * width_used * left_normal

        paired_centers = {row: 0.5 * (left_m + right_m) for row, left_m, right_m in pairs}

        reference = []
        count_pairs = 0
        count_singles = 0
        previous_row = None
        previous_center = None
        skipped_nonprogress = 0
        failure = None
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
                    termination = "END_OF_TRACKED_OBSERVATIONS"
                    break
                continue

            if not np.isfinite(center).all():
                termination = "NONFINITE_METRIC_CENTER"
                break

            u, v = ground_to_camera(self.inverse_homography, center)
            if not np.isfinite([u, v]).all() or not (0 <= u < W and TOP_GROUND_ROW <= v <= BOTTOM_GROUND_ROW):
                termination = "CENTER_LEFT_VISIBLE_IMAGE"
                break

            if not ground_is_usable(mask, u, v):
                termination = "CENTER_NOT_SEGMENTED_GROUND"
                break

            if previous_center is not None:
                row_gap = previous_row - row
                center_jump = float(np.linalg.norm(center - previous_center))
                forward_step = float(center[0] - previous_center[0])

                if row_gap > 9:
                    failure = f"ROW_GAP_{row_gap}_PX"
                    break

                if center_jump > 0.20:
                    failure = f"CENTER_JUMP_{center_jump:.3f}_M"
                    break

                if forward_step <= 0.000:
                    skipped_nonprogress += 1
                    if skipped_nonprogress > 2:
                        failure = f"REPEATED_NONPROGRESS_{forward_step:.4f}_M"
                        break
                    continue

            skipped_nonprogress = 0
            reference.append((float(center[0]), float(center[1])))
            previous_row = row
            previous_center = center

            if mode == "PAIR":
                count_pairs += 1
                color = (255, 0, 0)
            else:
                count_singles += 1
                color = (0, 140, 255)

            cv2.circle(debug, (round(u), round(v)), 2, color, -1)

        result["pairs"] = count_pairs
        result["single"] = count_singles
        result["width_m"] = round(width_used, 4) if width_used is not None else None
        result["width_source"] = width_source
        result["single_boundary_side"] = anchor_side.upper() if anchor_side is not None else "TRACKED_FROM_PAIR"
        result["termination_reason"] = termination

        if reference:
            near_x = reference[0][0]
            far_x = reference[-1][0]
            span = far_x - near_x
            result["near_x_m"] = round(near_x, 4)
            result["far_x_m"] = round(far_x, 4)
            result["observed_span_m"] = round(span, 4)
        else:
            near_x = far_x = span = None

        if failure is not None:
            result["status"] = "INVALID_REFERENCE_GEOMETRY"
            result["detail"] = failure
        elif not reference:
            result["status"] = "INVALID_NO_METRIC_REFERENCE"
            result["detail"] = "No center passed geometry checks"
        elif len(reference) < 6 or span < 0.08:
            result["status"] = "INVALID_INSUFFICIENT_SPAN"
            result["detail"] = f"{len(reference)} points; {span:.3f}m span"
        elif near_x > 0.45:
            result["status"] = "INVALID_NO_NEAR_COVERAGE"
            result["detail"] = f"Near corridor is unobserved ({near_x:.3f}m)"
        elif far_x >= 0.70 and span >= 0.25:
            result["status"] = "VALID_FOR_DIAGNOSTIC_TEST"
            result["detail"] = "Connected observed reference"
        else:
            result["status"] = "USABLE_SHORT_REFERENCE_DIAGNOSTIC"
            result["detail"] = f"Short path ends at {far_x:.3f}m"

        if result["status"] in ("VALID_FOR_DIAGNOSTIC_TEST", "USABLE_SHORT_REFERENCE_DIAGNOSTIC"):
            result["points_x_forward_y_left_m"] = [[round(x, 4), round(y, 4)] for x, y in reference]
            if len(tracked) >= 4:
                for side, index in (("left", 1), ("right", 2)):
                    trace = {row: values[index] for values in tracked for row in [values[0]] if values[index] is not None}
                    if len(trace) >= 4:
                        self.last_boundary_traces[side] = trace
                        self.last_boundary_times[side] = time.monotonic()

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