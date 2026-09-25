#!/usr/bin/env python3
# coding=utf-8

"""
autonomous_driving.py
=====================

ROS 2 line-following driver.

This version intentionally implements lane following only:
- No automatic overtaking
- No lane-changing commands
- LiDAR emergency stopping remains active
- Fixed BEV lookahead compensates for the physical camera blind spot
- Grace hold absorbs short segmentation failures
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from lane_analyzer import (
    LaneAnalyzer,
    STATE_BOTH,
    STATE_LEFT_ONLY,
    STATE_RIGHT_ONLY,
    STATE_DRIVABLE,
    DEFAULT_LANE_WIDTH_PX,
    DEFAULT_ROI_START_RATIO,
    DEFAULT_ROI_END_RATIO,
)


# =========================================================
# BEHAVIOR STATES
# =========================================================

# These constants are retained for future integration, but this version
# deliberately remains in STATE_LANE_KEEP.
STATE_LANE_KEEP = "LANE_KEEP"
STATE_CHANGING_LANE = "CHANGING_LANE"
STATE_PASSING = "PASSING"
STATE_RETURNING = "RETURNING"


# =========================================================
# DEFAULT PARAMETERS
# =========================================================

DEFAULT_MAX_SPEED = 0.25
DEFAULT_CENTER_OFFSET_PX = 30

# Fixed controller target. For a 160-pixel BEV:
# 0.82 * 159 ~= 130 pixels.
DEFAULT_CONTROL_TARGET_Y_RATIO = 0.82

# Number of temporarily missing frames that may reuse decayed steering.
DEFAULT_LOST_FRAME_LIMIT = 4

# Slow speed while using grace hold.
DEFAULT_GRACE_SPEED = 0.06

# Multiplicative decay applied to steering every lost frame.
DEFAULT_GRACE_OMEGA_DECAY = 0.70
DEFAULT_SAFETY_STOP_DISTANCE_M = 0.25


# =========================================================
# LOW-PASS FILTER
# =========================================================

class LowPassFilter:
    """Simple exponential low-pass filter."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.initialized = False
        self.current_value = 0.0

    def filter(self, value: float) -> float:
        value = float(value)

        if not self.initialized:
            self.current_value = value
            self.initialized = True
            return value

        self.current_value = (
            self.alpha * value + (1.0 - self.alpha) * self.current_value
        )

        return float(self.current_value)

    def reset(self) -> None:
        self.initialized = False
        self.current_value = 0.0


# =========================================================
# STEERING CONTROLLER
# =========================================================

class Controller:
    """Pixel-error PD steering controller."""

    def __init__(
        self,
        kp: float = 0.015,
        kd: float = 0.005,
        max_angular: float = 0.43,
    ):
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_angular = abs(float(max_angular))
        self.previous_error = 0.0

    def compute(self, error: float) -> float:
        error = float(error)

        derivative = error - self.previous_error
        self.previous_error = error

        omega = (self.kp * error + self.kd * derivative)

        return float(np.clip(omega, -self.max_angular, self.max_angular))

    def reset(self) -> None:
        self.previous_error = 0.0


# =========================================================
# DRIVER NODE
# =========================================================

class Driver(Node):
    """Vision-based lane-following driver."""

    def __init__(self):
        super().__init__("no_circle_driver")

        # ---------------------------------------------
        # Lane-analysis and controller parameters
        # ---------------------------------------------

        self.declare_parameter("default_lane_width", DEFAULT_LANE_WIDTH_PX)
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("center_offset_px", DEFAULT_CENTER_OFFSET_PX)
        self.declare_parameter("roi_start_ratio", DEFAULT_ROI_START_RATIO)
        self.declare_parameter("roi_end_ratio", DEFAULT_ROI_END_RATIO)
        self.declare_parameter("control_target_y_ratio", DEFAULT_CONTROL_TARGET_Y_RATIO)
        self.declare_parameter("lost_frame_limit", DEFAULT_LOST_FRAME_LIMIT)
        self.declare_parameter("grace_speed", DEFAULT_GRACE_SPEED)
        self.declare_parameter("grace_omega_decay", DEFAULT_GRACE_OMEGA_DECAY),
        self.declare_parameter("safety_stop_distance", DEFAULT_SAFETY_STOP_DISTANCE_M)
        default_lane_width = float(self.get_parameter("default_lane_width").value)
        self.max_speed = max(0.0, float(self.get_parameter("max_speed").value))
        self.center_offset_px = float(self.get_parameter("center_offset_px").value)
        roi_start_ratio = float(self.get_parameter("roi_start_ratio").value)
        roi_end_ratio = float(self.get_parameter("roi_end_ratio").value)
        self.control_target_y_ratio = float(np.clip(self.get_parameter("control_target_y_ratio").value, 0.0, 1.0,))
        self.lost_frame_limit = max(0, int(self.get_parameter("lost_frame_limit").value))
        self.grace_speed = max(0.0, float(self.get_parameter("grace_speed").value))
        self.grace_omega_decay = float(np.clip(self.get_parameter("grace_omega_decay").value, 0.0, 1.0))
        self.safety_stop_distance = max(0.0, float(self.get_parameter("safety_stop_distance").value))

        # ---------------------------------------------
        # Shared analyzer and controller
        # ---------------------------------------------

        self.bridge = CvBridge()

        self.lane_analyzer = LaneAnalyzer(
            lane_width_px=default_lane_width,
            camera_offset_x_px=0.0,
            roi_start_ratio=roi_start_ratio,
            roi_end_ratio=roi_end_ratio,
            alpha_lane_width=0.07,
        )

        self.ctrl = Controller()
        self.error_filter = LowPassFilter(alpha=0.25)

        # ---------------------------------------------
        # LiDAR state
        # ---------------------------------------------

        self.obstacle_dist = float("inf")
        self.left_clear = True
        self.right_clear = True

        # ---------------------------------------------
        # Lane-following state
        # ---------------------------------------------

        # Overtaking is intentionally frozen.
        self.behavior_state = STATE_LANE_KEEP
        self.overtake_side = None
        self.direction_sign = 0

        # Grace-hold memory.
        self.last_valid_center = None
        self.last_valid_error = 0.0
        self.last_valid_omega = 0.0
        self.lost_frames = 0

        # ---------------------------------------------
        # ROS interfaces
        # ---------------------------------------------

        self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            "/seg/bev_mask",
            self.cb,
            1,
        )

        self.pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            1,
        )

        self.get_logger().info(
            "Lane-following driver active. "
            "Overtaking disabled."
        )
        self.get_logger().info(
            f"Fixed control target ratio: "
            f"{self.control_target_y_ratio:.2f}"
        )
        self.get_logger().info(
            f"Grace hold: {self.lost_frame_limit} frames, "
            f"speed={self.grace_speed:.3f}, "
            f"decay={self.grace_omega_decay:.2f}"
        )

    # -----------------------------------------------------
    # LiDAR
    # -----------------------------------------------------

    def lidar_cb(self, msg: LaserScan) -> None:
        """Update front, left, and right obstacle measurements."""

        front_ranges = []
        left_ranges = []
        right_ranges = []

        for index, distance in enumerate(msg.ranges):
            angle = (
                msg.angle_min
                + index * msg.angle_increment
            )

            if not np.isfinite(distance):
                continue

            if not (
                msg.range_min
                < distance
                < msg.range_max
            ):
                continue

            if -0.26 <= angle <= 0.26:
                front_ranges.append(distance)

            elif 0.52 <= angle <= 1.31:
                left_ranges.append(distance)

            elif -1.31 <= angle <= -0.52:
                right_ranges.append(distance)

        self.obstacle_dist = (
            min(front_ranges)
            if front_ranges
            else float("inf")
        )

        self.left_clear = (
            min(left_ranges) > 0.35
            if left_ranges
            else True
        )

        self.right_clear = (
            min(right_ranges) > 0.35
            if right_ranges
            else True
        )

    # -----------------------------------------------------
    # Grace hold
    # -----------------------------------------------------

    def _publish_grace_or_stop(
        self,
        reason: str,
    ) -> None:
        """
        Reuse decayed steering for a short loss window, then stop.
        This method is only called when the current tracking result is invalid.
        """

        self.lost_frames += 1

        cmd = Twist()

        grace_available = (self.last_valid_center is not None and self.lost_frames <= self.lost_frame_limit)

        if grace_available:
            self.last_valid_omega *= self.grace_omega_decay

            cmd.linear.x = min(self.grace_speed, self.max_speed)
            cmd.angular.z = float(np.clip(self.last_valid_omega, -self.ctrl.max_angular, self.ctrl.max_angular))

            self.get_logger().warn(
                f"{reason}; grace hold "
                f"{self.lost_frames}/"
                f"{self.lost_frame_limit}"
            )

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            self.get_logger().warn(
                f"{reason}; stopping after "
                f"{self.lost_frames} lost frames"
            )

        self.pub.publish(cmd)

    # -----------------------------------------------------
    # Main BEV callback
    # -----------------------------------------------------

    def cb(self, msg: Image) -> None:
        """Process one BEV segmentation mask."""

        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8",
            )
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        if mask is None or mask.ndim != 2:
            self._publish_grace_or_stop("Invalid BEV mask")
            return

        height, width = mask.shape

        # ---------------------------------------------
        # Emergency LiDAR stop
        # ---------------------------------------------

        if (self.obstacle_dist < self.safety_stop_distance):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            self.get_logger().warn(f"Obstacle ahead: {self.obstacle_dist:.2f} m")
            self.pub.publish(cmd)
            return

        # ---------------------------------------------
        # Shared lane analysis
        # ---------------------------------------------

        _, _, center_path, _, _, state, tune_offset_px = self.lane_analyzer.analyze(mask)

        path_valid = (center_path is not None and len(center_path) >= 2 and np.isfinite(center_path).all())

        if not path_valid:
            self._publish_grace_or_stop("No valid tracking path")
            return

        # ---------------------------------------------
        # Fixed lookahead control
        # ---------------------------------------------

        target_y = (self.control_target_y_ratio * float(height - 1))
        distances = np.abs(center_path[:, 1] - target_y)
        closest_index = int(np.argmin(distances))
        center_x = float(center_path[closest_index, 0])
        selected_y = float(center_path[closest_index, 1])

        # The extrapolated path should pass near the fixed control row.
        # If it does not, the current path is not useful for control.
        maximum_y_error = max(8.0, height * 0.12)

        target_y_error = abs(selected_y - target_y)

        center_plausible = (np.isfinite(center_x) and -float(width) <= center_x <= 2.0 * float(width))

        if (target_y_error > maximum_y_error or not center_plausible):
            self._publish_grace_or_stop("Path does not reach a valid control target")
            return

        # ---------------------------------------------
        # Steering calculation
        # ---------------------------------------------
        if state == STATE_BOTH:
            desired_center = width / 2.0
        else:
            # Apply the tuning offset directly to the target center
            desired_center = (width / 2.0) + tune_offset_px

        raw_error = desired_center - center_x
        smoothed_error = self.error_filter.filter(raw_error)
        omega = self.ctrl.compute(smoothed_error)

        # ---------------------------------------------
        # Save valid tracking memory
        # ---------------------------------------------

        self.last_valid_center = center_x
        self.last_valid_error = raw_error
        self.last_valid_omega = omega
        self.lost_frames = 0

        # ---------------------------------------------
        # Speed selection
        # ---------------------------------------------

        speed = self.max_speed - abs(omega) * 0.08

        if state in (STATE_LEFT_ONLY, STATE_RIGHT_ONLY, STATE_DRIVABLE):
            speed = min(speed, self.max_speed * 0.70)

        minimum_tracking_speed = min(0.10, self.max_speed)

        speed = float(np.clip(speed, minimum_tracking_speed, self.max_speed))

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega

        self.pub.publish(cmd)


# =========================================================
# MAIN
# =========================================================

def main(args=None) -> None:
    rclpy.init(args=args)

    node = Driver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()