#!/usr/bin/env python3
"""Stationary test of homography outside the fitted trapezoid.

Inputs:  /seg/calibration_grid (320x240 BGR camera image)
         /seg/mask_raw         (320x240 mono8 segmentation classes)
Output:  /seg/metric_calibration_debug
No motion-command publishers.
"""

import random

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


WIDTH, HEIGHT = 320, 240

# BL, BR, TR, TL: camera pixels.
SRC = np.float32([
    [11.0, 238.0],
    [305.5, 238.0],
    [210.0, 160.0],
    [105.0, 160.0],
])

# Matching ground positions: X forward, Y left, in meters.
DST = np.float32([
    [0.305, +0.210],
    [0.305, -0.210],
    [0.930, -0.2115],
    [0.930, +0.2115],
])

# Check these against your segmentation model's class IDs.
ROAD, WHITE, YELLOW = 1, 2, 3


class FullGroundCalibrationTest(Node):
    def __init__(self):
        super().__init__("test_metric_calibration_full_ground")
        self.bridge = CvBridge()

        self.declare_parameter("seed", 17)
        self.declare_parameter("u", -1)
        self.declare_parameter("v", -1)

        self.seed = int(self.get_parameter("seed").value)
        manual_u = int(self.get_parameter("u").value)
        manual_v = int(self.get_parameter("v").value)

        if (manual_u == -1) != (manual_v == -1):
            raise ValueError("Provide both u and v, or neither")

        self.manual_point = (
            None if manual_u == -1 else (manual_u, manual_v)
        )

        self.polygon = np.rint(SRC).astype(np.int32)
        self.homography = cv2.getPerspectiveTransform(SRC, DST)

        self.latest_image = None
        self.latest_mask = None
        self.selected_point = None
        self.predicted_meters = None
        self.last_report = None

        self.create_subscription(
            Image, "/seg/calibration_grid", self.image_cb, 1
        )
        self.create_subscription(
            Image, "/seg/mask_raw", self.mask_cb, 1
        )
        self.debug_pub = self.create_publisher(
            Image, "/seg/metric_calibration_debug", 1
        )
        self.create_timer(0.2, self.publish_debug)

        self.get_logger().info(
            "Stationary test only. No /cmd_vel or "
            "/cmd_vel_request publisher."
        )

    def image_cb(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as exc:
            self.get_logger().error(f"Calibration image error: {exc}")
            return

        if image.shape[:2] != (HEIGHT, WIDTH):
            self.get_logger().error(
                f"Calibration image is {image.shape[1]}x"
                f"{image.shape[0]}; expected {WIDTH}x{HEIGHT}"
            )
            return

        self.latest_image = (image.copy(), msg.header)

    def mask_cb(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8"
            )
        except Exception as exc:
            self.get_logger().error(f"Mask error: {exc}")
            return

        if mask.shape != (HEIGHT, WIDTH):
            self.get_logger().error(
                f"Mask is {mask.shape[::-1]}; expected "
                f"{WIDTH}x{HEIGHT}"
            )
            return

        self.latest_mask = mask.copy()

    def eligible(self, mask, u, v):
        if not (0 <= u < WIDTH and 60 <= v < HEIGHT):
            return False

        # Strictly outside, not on the calibration boundary.
        if cv2.pointPolygonTest(
            self.polygon, (float(u), float(v)), False
        ) >= 0:
            return False

        return int(mask[v, u]) in (ROAD, WHITE, YELLOW)

    def select_point(self):
        mask = self.latest_mask
        if mask is None:
            return

        if self.manual_point is not None:
            u, v = self.manual_point
            if not self.eligible(mask, u, v):
                message = (
                    f"Custom pixel ({u},{v}) rejected: it must be "
                    "outside the trapezoid, at v>=60, within the image, "
                    "and classified as ground/marking. Check the mask "
                    "and choose another floor point."
                )
                if message != self.last_report:
                    self.get_logger().warn(message)
                    self.last_report = message
                return
        else:
            ground = np.isin(mask, (ROAD, WHITE, YELLOW))
            candidates = []

            for v in range(60, HEIGHT):
                for u in range(WIDTH):
                    if ground[v, u] and self.eligible(mask, u, v):
                        candidates.append((u, v))

            if not candidates:
                message = "No eligible ground pixels outside trapezoid"
                if message != self.last_report:
                    self.get_logger().warn(message)
                    self.last_report = message
                return

            u, v = random.Random(self.seed).choice(candidates)

        point = np.float32([[[float(u), float(v)]]])
        mapped = cv2.perspectiveTransform(
            point, self.homography
        )[0, 0]

        if not np.isfinite(mapped).all():
            self.get_logger().warn(
                f"Non-finite result for pixel ({u},{v})"
            )
            return

        self.selected_point = (u, v)
        self.predicted_meters = (
            float(mapped[0]), float(mapped[1])
        )

        x_forward, y_left = self.predicted_meters
        self.get_logger().info(
            f"OUTSIDE pixel ({u},{v}) -> predicted "
            f"X_forward={x_forward:.4f} m, "
            f"Y_left={y_left:+.4f} m. "
            "Measure the floor point at the crosshair."
        )

    def publish_debug(self):
        if self.latest_image is None or self.latest_mask is None:
            return

        if self.selected_point is None:
            self.select_point()
        if self.selected_point is None:
            return

        image, header = self.latest_image
        debug = image.copy()
        u, v = self.selected_point
        x_forward, y_left = self.predicted_meters

        # Selected ground point: full-image red crosshair.
        cv2.line(debug, (0, v), (WIDTH - 1, v), (0, 0, 255), 1)
        cv2.line(debug, (u, 0), (u, HEIGHT - 1), (0, 0, 255), 1)
        cv2.circle(debug, (u, v), 3, (0, 0, 0), -1)

        # Blue image midpoint is a visual guide, NOT a calibrated
        # projection of the vehicle's physical centerline.
        cv2.line(
            debug, (WIDTH // 2, 0),
            (WIDTH // 2, HEIGHT - 1),
            (255, 0, 0), 1
        )

        cv2.putText(
            debug, f"OUTSIDE ({u},{v})", (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1
        )
        cv2.putText(
            debug, f"X forward {x_forward:.3f} m", (6, 31),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1
        )
        cv2.putText(
            debug, f"Y left {y_left:+.3f} m", (6, 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1
        )

        output = self.bridge.cv2_to_imgmsg(
            debug, encoding="bgr8"
        )
        output.header = header
        self.debug_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = FullGroundCalibrationTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()