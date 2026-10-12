#!/usr/bin/env python3
"""Reference test only: compare two-boundary and one-boundary BEV centers.

Publishes /seg/reference_debug. Never publishes /cmd_vel.
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


BG, ROAD, WHITE, YELLOW, VEHICLE = 0, 1, 2, 3, 4

COLORS = np.array(
    [
        [0, 255, 0],       # background
        [100, 100, 100],   # road
        [255, 255, 255],   # white
        [0, 255, 255],     # yellow
        [0, 0, 255],       # vehicle
    ],
    dtype=np.uint8,
)


def components_for_class(mask: np.ndarray, class_id: int):
    binary = (mask == class_id).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    minimum_area = max(12, int(mask.size * 0.0003))
    valid_ids = {
        label_id
        for label_id in range(1, count)
        if stats[label_id, cv2.CC_STAT_AREA] >= minimum_area
    }
    return labels, valid_ids


def candidates_at_row(label_sets, row: int, band: int, height: int):
    """Find physical marking candidates near a row, merging nearby colors."""
    y0 = max(0, row - band)
    y1 = min(height, row + band + 1)

    markings = []  # (median_x, pixel_count)

    for labels, valid_ids, _class_id in label_sets:
        local = labels[y0:y1]

        for label_id in valid_ids:
            _, xs = np.where(local == label_id)
            if xs.size >= 3:
                markings.append((float(np.median(xs)), int(xs.size)))

    if not markings:
        return []

    markings.sort(key=lambda item: item[0])
    width = label_sets[0][0].shape[1]

    # At 160 px, merge markings whose median X positions differ by <= 10 px.
    # This is intended to combine white/yellow pixels on ONE boundary,
    # not two lane boundaries. Inspect the result before changing this value.
    merge_distance = min(10.0, max(3.0, 0.06 * width))

    groups = []
    current = [markings[0]]

    for marking in markings[1:]:
        if marking[0] - current[-1][0] <= merge_distance:
            current.append(marking)
        else:
            groups.append(current)
            current = [marking]

    groups.append(current)

    candidates = []
    for index, group in enumerate(groups):
        xs = np.array([item[0] for item in group], dtype=np.float64)
        weights = np.array([item[1] for item in group], dtype=np.float64)
        physical_x = float(np.average(xs, weights=weights))

        # The key is no longer used to identify a color component across
        # rows. The one-boundary selection below uses spatial continuity.
        candidates.append((physical_x, ("boundary", index)))

    return candidates


def measured_rows(mask: np.ndarray):
    """
    Track corridor boundaries from near to far.
    
    Anchors to boundaries closest to the robot near the bumper and tracks them
    upward using spatial continuity, preventing pairing with adjacent corridors.
    """
    height, width = mask.shape
    robot_x = width / 2.0

    white_labels, white_ids = components_for_class(mask, WHITE)
    yellow_labels, yellow_ids = components_for_class(mask, YELLOW)

    label_sets = [
        (white_labels, white_ids, WHITE),
        (yellow_labels, yellow_ids, YELLOW),
    ]

    step = max(3, round(height / 55))
    band = max(1, round(height / 160))
    min_gap = 0.12 * width
    max_gap = 0.95 * width
    max_step_dx = max(10.0, 0.08 * width) + 0.65 * step

    sampled_rows = list(
        range(
            min(height - 1, round(0.96 * height)),
            max(0, round(0.15 * height)),
            -step,
        )
    )

    # Collect merged physical candidates at each sampled row
    row_candidates = {}
    for y in sampled_rows:
        row_candidates[y] = candidates_at_row(label_sets, y, band, height)

    # 1. Identify near-field boundary anchors closest to the robot
    left_track_x = None
    right_track_x = None

    for y in sampled_rows[:6]:
        candidates = [c[0] for c in row_candidates[y]]
        lefts = [x for x in candidates if x < robot_x]
        rights = [x for x in candidates if x > robot_x]

        if left_track_x is None and lefts:
            left_track_x = max(lefts)
        if right_track_x is None and rights:
            right_track_x = min(rights)

        if left_track_x is not None or right_track_x is not None:
            break

    # 2. Track boundaries from near to far
    paired = []
    unpaired = []

    for y in sampled_rows:
        candidates = [c[0] for c in row_candidates[y]]
        if not candidates:
            continue

        matched_left = None
        matched_right = None

        if left_track_x is not None:
            close_left = [x for x in candidates if abs(x - left_track_x) <= max_step_dx]
            if close_left:
                matched_left = min(close_left, key=lambda x: abs(x - left_track_x))

        if right_track_x is not None:
            close_right = [x for x in candidates if abs(x - right_track_x) <= max_step_dx]
            if close_right:
                matched_right = min(close_right, key=lambda x: abs(x - right_track_x))

        # Check if both boundaries form a valid corridor pair
        if matched_left is not None and matched_right is not None and matched_left < matched_right:
            gap = matched_right - matched_left
            if min_gap <= gap <= max_gap:
                paired.append(
                    (y, matched_left, (matched_left + matched_right) / 2.0, matched_right, ("boundary", 0), ("boundary", 1))
                )
                left_track_x = matched_left
                right_track_x = matched_right
                continue

        # Single boundary continuation
        if matched_left is not None and matched_right is None:
            unpaired.append((y, matched_left, ("boundary", 0)))
            left_track_x = matched_left
        elif matched_right is not None and matched_left is None:
            unpaired.append((y, matched_right, ("boundary", 1)))
            right_track_x = matched_right

    return paired, unpaired


class ReferencePathTest(Node):
    def __init__(self):
        super().__init__("reference_path_test")
        self.bridge = CvBridge()

        # Learned from sufficiently consistent, paired observations.
        # Not reset when a boundary briefly disappears.
        self.learned_width_px = None

        self.create_subscription(Image, "/seg/bev_mask", self.on_mask, 1)
        self.debug_pub = self.create_publisher(
            Image, "/seg/reference_debug", 1
        )

        self.get_logger().info(
            "REFERENCE TEST ONLY: publishes images; never publishes /cmd_vel"
        )

    def on_mask(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8"
            )
        except Exception as exc:
            self.get_logger().error(f"BEV conversion failed: {exc}")
            return

        if mask.ndim != 2:
            self.get_logger().error("Expected a mono8 BEV class mask")
            return

        height, width = mask.shape
        debug = COLORS[
            np.clip(mask.astype(np.int16), 0, len(COLORS) - 1)
        ].copy()

        paired_raw, unpaired_raw = measured_rows(mask)
        step = max(3, round(height / 55))

        def near_contiguous(rows):
            """Keep the near-to-far run; do not bridge a missing section."""
            if not rows:
                return []

            result = [rows[0]]
            for row in rows[1:]:
                if result[-1][0] - row[0] > 3 * step:
                    break
                result.append(row)
            return result

        # A couple of distant pairs must not be treated as a second path.
        paired_run = near_contiguous(paired_raw)
        paired = paired_run if len(paired_run) >= 8 else []

        measured_gap = None
        current_width = None

        if paired:
            gaps = np.asarray(
                [row[3] - row[1] for row in paired],
                dtype=np.float64,
            )
            measured_gap = float(np.median(gaps))
            median_deviation = float(
                np.median(np.abs(gaps - measured_gap))
            )

            if median_deviation <= max(8.0, 0.12 * measured_gap):
                current_width = measured_gap
                if self.learned_width_px is None:
                    self.learned_width_px = current_width
                else:
                    self.learned_width_px = (
                        0.8 * self.learned_width_px
                        + 0.2 * current_width
                    )

        width_used = (
            current_width
            if current_width is not None
            else self.learned_width_px
        )
        width_source = (
            "CURRENT" if current_width is not None
            else "LEARNED" if width_used is not None
            else "UNKNOWN"
        )

        # Keep the spatial-tracking approach from your latest version.
        # unpaired_raw is ordered from near (large Y) to far (small Y).
        selected_rows = []
        tracked_side = None
        previous_x = None
        previous_y = None

        for y, boundary_x, _key in unpaired_raw:
            if previous_x is None:
                selected_rows.append((y, boundary_x))
                previous_x = boundary_x
                previous_y = y
                tracked_side = (
                    "LEFT" if boundary_x < width / 2.0 else "RIGHT"
                )
                continue

            row_distance = previous_y - y
            if row_distance > 3 * step:
                break

            max_horizontal_change = (
                max(9.0, 0.08 * width)
                + 0.65 * row_distance
            )
            if abs(boundary_x - previous_x) > max_horizontal_change:
                break

            selected_rows.append((y, boundary_x))
            previous_x = boundary_x
            previous_y = y

        # Draw only the accepted contiguous paired run.
        for y, left_x, center_x, right_x, _, _ in paired:
            cv2.circle(debug, (round(left_x), y), 2, (255, 0, 255), -1)
            cv2.circle(debug, (round(right_x), y), 2, (255, 255, 0), -1)
            cv2.circle(debug, (round(center_x), y), 2, (255, 0, 0), -1)

        # Store the points actually available, not merely candidate rows.
        single_points = []  # (y, estimated_center_x)

        for y, boundary_x in selected_rows:
            # White indicates the marking used for the estimate.
            cv2.circle(
                debug, (round(boundary_x), y), 2,
                (255, 255, 255), -1
            )

            if width_used is None or tracked_side is None:
                continue

            direction = 1.0 if tracked_side == "LEFT" else -1.0
            estimated_x = boundary_x + direction * width_used / 2.0

            # Stop at the first off-image estimate. Do not join dots
            # on opposite sides of an unavailable section.
            if not (0 <= estimated_x < width):
                break

            single_points.append((y, estimated_x))
            cv2.circle(
                debug, (round(estimated_x), y), 2,
                (0, 140, 255), -1
            )

        # The marker is an assumed robot origin, not observed ground.
        cv2.circle(
            debug, (width // 2, height - 1), 4,
            (0, 0, 255), -1
        )

        paired_near_x = paired[0][2] if paired else None
        single_near_x = (
            single_points[0][1] if single_points else None
        )

        # Check whether a blue section and a farther-ahead orange
        # section actually join. Compare adjacent endpoints, not
        # PAIR_NEAR_X and SINGLE_NEAR_X (different image rows).
        joins = False
        if paired and single_points:
            pair_far_y, _, pair_far_x, _, _, _ = paired[-1]
            single_near_y, single_near_center_x = single_points[0]

            y_gap = pair_far_y - single_near_y
            allowed_x_gap = (
                max(12.0, 0.08 * width)
                + 0.8 * max(0, y_gap)
            )
            joins = (
                0 <= y_gap <= 3 * step
                and abs(pair_far_x - single_near_center_x)
                <= allowed_x_gap
            )

        # Determine the near-to-far coverage of ONE usable path.
        # An isolated far-away pair cannot extend this span.
        usable = []

        if paired:
            usable.extend((row[0], row[2]) for row in paired)

        if single_points and (not paired or joins):
            usable.extend(single_points)

        verdict = "VALID_REFERENCE"

        if not usable:
            verdict = "INVALID_NO_PATH"
        elif paired and single_points and not joins:
            verdict = "INVALID_DISCONNECTED"
        else:
            near_y = max(point[0] for point in usable)
            far_y = min(point[0] for point in usable)
            span = near_y - far_y

            if near_y < 0.82 * height:
                verdict = "INVALID_NO_NEAR_COVERAGE"
            elif far_y > 0.70 * height or span < 0.25 * height:
                verdict = "INVALID_SHORT_LOOKAHEAD"

        if usable:
            near_y = max(point[0] for point in usable)
            far_y = min(point[0] for point in usable)
            span = near_y - far_y
        else:
            near_y = far_y = span = None

        def fmt(value):
            return "NA" if value is None else f"{value:.0f}"

        print(
            f"[reference_path_test] "
            f"PAIRS_RAW {len(paired_raw)} PAIRS {len(paired)} "
            f"SINGLE {len(single_points)} "
            f"PAIR_NEAR_X {fmt(paired_near_x)} "
            f"SINGLE_NEAR_X {fmt(single_near_x)} "
            f"GAP {fmt(measured_gap)}px "
            f"WIDTH_USED {fmt(width_used)}px "
            f"WIDTH_SOURCE {width_source} "
            f"SINGLE_SIDE {tracked_side or 'NA'} "
            f"NEAR_Y {fmt(near_y)} FAR_Y {fmt(far_y)} "
            f"SPAN {fmt(span)}px "
            f"JOIN {'YES' if joins else 'NO'} "
            f"STATUS {verdict}",
            flush=True,
        )

        output = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        output.header = msg.header
        self.debug_pub.publish(output)


def main():
    rclpy.init()
    node = ReferencePathTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()