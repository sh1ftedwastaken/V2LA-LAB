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
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from scipy.optimize import minimize

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

# Number of temporarily missing frames that may reuse decayed steering.
DEFAULT_LOST_FRAME_LIMIT = 4

# Slow speed while using grace hold.
DEFAULT_GRACE_SPEED = 0.06

# Multiplicative decay applied to steering every lost frame.
DEFAULT_GRACE_OMEGA_DECAY = 0.70
DEFAULT_SAFETY_STOP_DISTANCE_M = 0.25


class MPCController:
    """
    MPC: Optimize a sequence of normalized velocities (v_norm) and steering commands (omega)
    to minimize the error with respect to the future trajectory (center_path) while maximizing speed.
    """
    def __init__(self, N=10, dt=0.1, v_pixel_max=120.0, max_angular=0.45):
        self.N = N
        self.dt = dt
        self.v_pixel_max = v_pixel_max  # Maximum simulated speed in pixels/sec
        self.max_omega = abs(float(max_angular))

        # Weights (Tuning)
        self.Q = 10.0     # Cost for deviating from the blue line
        self.R = 10.0    # Control effort (Prevents unnecessary steering) [4]
        self.Rd = 20.0   # Steering smoothness (Prevents jerky wheel movements)
        self.Qv = 10.0   # Penalty for going slow (Rewards driving at v_norm = 1.0)
        self.Rdv = 15.0  # Acceleration smoothness (Prevents sudden braking/accelerating)

        # Initial guess: N velocities (default 1.0) + N omegas (default 0.0)
        self.prev_controls = np.zeros(2 * self.N)
        self.prev_controls[:self.N] = 1.0  # Start with a guess of max speed

    def predict_trajectory(self, x0, y0, theta0, vs_norm, omegas):
        """Simulate where the car will be using dynamic velocities and steering."""
        x, y, theta = x0, y0, theta0
        trajectory = np.zeros((self.N, 2))
        for i in range(self.N):
            # Scale the normalized velocity [0.0 to 1.0] to pixel speed
            v_pixel = vs_norm[i] * self.v_pixel_max
            x += v_pixel * np.cos(theta) * self.dt
            y += v_pixel * np.sin(theta) * self.dt
            theta -= omegas[i] * self.dt
            trajectory[i] = [x, y]
        return trajectory

    def cost_function(self, controls, x0, y0, theta0, target_trajectory):
        """Calculate total cost using perfectly smooth Euclidean distance."""
        vs_norm = controls[:self.N]
        omegas = controls[self.N:]

        trajectory = self.predict_trajectory(x0, y0, theta0, vs_norm, omegas)
        cost = 0.0

        for i in range(self.N):
            # 1. Smooth Tracking Error (Perfect Parabola, no np.interp failure)
            # Distance squared between where the car is predicted to be, and the target waypoint
            dist_sq = (trajectory[i, 0] - target_trajectory[i, 0])**2 + \
                      (trajectory[i, 1] - target_trajectory[i, 1])**2
            
            cost += self.Q * dist_sq

            # 2. Steering effort & smoothness
            cost += self.R * (omegas[i]**2)
            if i > 0:
                cost += self.Rd * ((omegas[i] - omegas[i-1])**2)

            # 3. Velocity reward & smoothness
            cost += self.Qv * ((1.0 - vs_norm[i])**2)
            if i > 0:
                cost += self.Rdv * ((vs_norm[i] - vs_norm[i-1])**2)

        return cost

    def compute(self, x0, y0, theta0, ref_path):
        """Solve for the optimal velocity and steering commands."""
        
        # --- NEW: Pre-calculate N target waypoints along the blue line ---
        # Calculate the cumulative distance along the blue line
        diffs = np.diff(ref_path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.zeros(len(ref_path))
        cum_dists[1:] = np.cumsum(dists)

        target_trajectory = np.zeros((self.N, 2))
        for i in range(self.N):
            # Find the point on the blue line that the car SHOULD reach at step i
            target_dist = (i + 1) * (self.v_pixel_max * self.dt)
            
            # Find the closest index on the blue line
            idx = np.searchsorted(cum_dists, target_dist)
            if idx >= len(ref_path):
                idx = len(ref_path) - 1
                
            target_trajectory[i] = ref_path[idx]
        # -----------------------------------------------------------------

        bounds = [(0.2, 1.0) for _ in range(self.N)] + \
                 [(-self.max_omega, self.max_omega) for _ in range(self.N)]

        result = minimize(
            self.cost_function,
            self.prev_controls,
            args=(x0, y0, theta0, target_trajectory), # Pass the fixed targets here!
            method='L-BFGS-B',
            bounds=bounds
        )

        optimal_controls = result.x

        self.prev_controls[:self.N-1] = optimal_controls[1:self.N]
        self.prev_controls[self.N-1] = optimal_controls[self.N-1]
        self.prev_controls[self.N:2*self.N-1] = optimal_controls[self.N+1:]
        self.prev_controls[-1] = optimal_controls[-1]

        best_v_norm = float(optimal_controls[0])
        best_omega = float(optimal_controls[self.N])

        predicted_traj = self.predict_trajectory(
            x0, y0, theta0, 
            optimal_controls[:self.N], 
            optimal_controls[self.N:]
        )

        return best_v_norm, best_omega, predicted_traj


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
        self.declare_parameter("roi_start_ratio", DEFAULT_ROI_START_RATIO)
        self.declare_parameter("roi_end_ratio", DEFAULT_ROI_END_RATIO)
        self.declare_parameter("lost_frame_limit", DEFAULT_LOST_FRAME_LIMIT)
        self.declare_parameter("grace_speed", DEFAULT_GRACE_SPEED)
        self.declare_parameter("grace_omega_decay", DEFAULT_GRACE_OMEGA_DECAY)
        self.declare_parameter("safety_stop_distance", DEFAULT_SAFETY_STOP_DISTANCE_M)
        default_lane_width = float(self.get_parameter("default_lane_width").value)
        self.max_speed = max(0.0, float(self.get_parameter("max_speed").value))
        roi_start_ratio = float(self.get_parameter("roi_start_ratio").value)
        roi_end_ratio = float(self.get_parameter("roi_end_ratio").value)
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

        # Initialize MPC
        self.mpc = MPCController(N=10, dt=0.1, v_pixel_max=120.0, max_angular=0.45)
        
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
            LaserScan, "/scan", self.lidar_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, "/seg/bev_mask", self.cb, 1
        )
        self.pub = self.create_publisher(
            Twist, "/cmd_vel", 1
        )
        self.pub_mpc_traj = self.create_publisher(
            Float32MultiArray, "/seg/mpc_trajectory", 1
        )

        self.get_logger().info(
            "Lane-following driver active. "
            "Overtaking disabled."
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
            min(front_ranges) if front_ranges else float("inf")
        )

        self.left_clear = (
            min(left_ranges) > 0.35 if left_ranges else True
        )

        self.right_clear = (
            min(right_ranges) > 0.35 if right_ranges else True
        )

    # -----------------------------------------------------
    # HELPERS
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
            cmd.angular.z = float(np.clip(self.last_valid_omega, -self.mpc.max_omega, self.mpc.max_omega))

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
        
    def _publish_mpc_debug(self, predicted_traj, v_norm, omega, state):
        if predicted_traj is not None:
            traj_msg = Float32MultiArray()
            traj_msg.data = predicted_traj.flatten().tolist()  # Flatten Nx2 array to 1D list
            self.pub_mpc_traj.publish(traj_msg)

        # Calculate final speed
        speed = self.max_speed * v_norm
        if state in (STATE_LEFT_ONLY, STATE_RIGHT_ONLY, STATE_DRIVABLE):
            speed = min(speed, self.max_speed * 0.70)
        speed = float(np.clip(speed, 0.10, self.max_speed))

        # Print to the terminal
        self.get_logger().info(f"MPC | v_norm: {v_norm:.2f} | Speed: {speed:.2f} m/s | Omega: {omega:.2f} rad/s")

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

        _, _, center_path, center_y_min, center_y_max, state, tune_offset_px = self.lane_analyzer.analyze(mask)

        path_valid = (center_path is not None and len(center_path) >= 2 and np.isfinite(center_path).all())

        if not path_valid:
            self._publish_grace_or_stop("No valid tracking path")
            return
        
        # ---------------------------------------------
        # MPC Dynamic Velocity and Steering Calculation
        # ---------------------------------------------
        
        # The car always physically starts at the exact center of the bumper
        car_x = width / 2.0
        car_y = float(height)
        car_theta = -np.pi / 2.0 

        # Instead of shifting the car, we shift the mathematical target path!
        # This keeps the red line perfectly anchored to the car's bumper on the screen.
        solver_path = center_path.copy()
        if state != STATE_BOTH:
            # Note: We ADD the offset to the path, which is mathematically 
            # equivalent to SUBTRACTING it from the car
            solver_path[:, 0] += tune_offset_px

        # The MPC now returns optimal velocity, steering, AND its predicted path
        v_norm, omega, predicted_traj = self.mpc.compute(car_x, car_y, car_theta, solver_path)
        
        # Publish debug information about the MPC to BEV
        self._publish_mpc_debug(predicted_traj, v_norm, omega, state)
        
        # ---------------------------------------------
        # Save memory for grace_hold
        # ---------------------------------------------
        self.last_valid_omega = omega
        self.lost_frames = 0

        # --- NEW: Dynamic speed based on MPC prediction ---
        # The fixed formula `speed = self.max_speed - abs(omega) * 0.08` has been removed.
        speed = self.max_speed * v_norm
        
        # Optional: Maintain a safety limit if the camera is missing a line
        if state in (STATE_LEFT_ONLY, STATE_RIGHT_ONLY, STATE_DRIVABLE):
            speed = min(speed, self.max_speed * 0.70)
            
        speed = float(np.clip(speed, 0.10, self.max_speed))

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