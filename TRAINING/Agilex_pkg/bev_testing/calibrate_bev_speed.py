#!/usr/bin/env python3
"""
calibrate_bev_speed.py
======================
Calibrate v_pixel_max by running timed forward bursts at fixed speed
and measuring the vertical pixel displacement on /seg/bev_mask.
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class SpeedCalibrator(Node):
    def __init__(self):
        super().__init__("calibrate_bev_speed")
        self.bridge = CvBridge()

        self.declare_parameter("speed", 0.10)       # Command speed in m/s
        self.declare_parameter("duration", 1.5)    # Drive duration in seconds

        self.cmd_speed = float(self.get_parameter("speed").value)
        self.duration = float(self.get_parameter("duration").value)

        self.latest_mask = None
        self.sub = self.create_subscription(Image, "/seg/bev_mask", self.mask_cb, 1)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.get_logger().info("Speed Calibrator Initialized.")
        self.get_logger().info(f"Target speed: {self.cmd_speed} m/s | Duration: {self.duration} s")
        self.get_logger().info("Waiting for /seg/bev_mask frames...")

        # Timer to start sequence once the camera stream is ready
        self.timer = self.create_timer(1.0, self.run_calibration)
        self.started = False

    def mask_cb(self, msg: Image):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion error: {exc}")

    def stop_car(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        for _ in range(5):
            self.pub.publish(stop_cmd)
            time.sleep(0.02)

    def run_calibration(self):
        if self.started:
            return
        if self.latest_mask is None:
            return

        self.started = True
        self.timer.cancel()

        self.get_logger().info("Capturing initial BEV mask before motion...")
        time.sleep(0.5)
        initial_mask = self.latest_mask.copy().astype(np.float32)

        self.get_logger().info(f"Driving forward at {self.cmd_speed} m/s for {self.duration} seconds...")
        cmd = Twist()
        cmd.linear.x = self.cmd_speed
        cmd.angular.z = 0.0

        start_time = time.time()
        while time.time() - start_time < self.duration:
            self.pub.publish(cmd)
            time.sleep(0.02)

        self.stop_car()
        self.get_logger().info("Car stopped. Waiting for image to settle...")
        time.sleep(0.6)

        final_mask = self.latest_mask.copy().astype(np.float32)

        # Calculate translation shift using 2D phase correlation
        shift, response = cv2.phaseCorrelate(initial_mask, final_mask)
        dx, dy = shift

        # In BEV coordinates, forward travel moves features downward (+Y shift)
        pixel_shift_y = abs(dy)
        pixels_per_second = pixel_shift_y / self.duration

        # Full-throttle normalized speed (1.0) scaling
        v_pixel_max_est = pixels_per_second / (self.cmd_speed / 0.25)

        print("\n" + "=" * 55)
        print("               CALIBRATION RESULTS               ")
        print("=" * 55)
        print(f"Commanded Speed : {self.cmd_speed:.2f} m/s")
        print(f"Travel Duration : {self.duration:.2f} s")
        print(f"Shift (dx, dy)  : ({dx:.2f} px, {dy:.2f} px)")
        print(f"Vertical Shift  : {pixel_shift_y:.2f} pixels")
        print(f"Observed Rate   : {pixels_per_second:.2f} px/s")
        print(f"Estimated v_pixel_max (at max 0.25 m/s): {v_pixel_max_est:.2f}")
        print("=" * 55 + "\n")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SpeedCalibrator()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.stop_car()
        node.destroy_node()


if __name__ == "__main__":
    main()