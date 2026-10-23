#!/usr/bin/env python3
"""Read-only camera-derived metric lane-state diagnostic.

Input:
    /lane/metric_reference_debug  std_msgs/String containing JSON

Output:
    /lane/lane_state_debug        std_msgs/String containing JSON

Never publishes /cmd_vel or /cmd_vel_request.
No odometry or IMU prediction yet.
"""

import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


REFERENCE_TOPIC = "/lane/metric_reference"
STATE_TOPIC = "/lane/lane_state_debug"

VALID_STATUSES = ("VALID_FOR_DIAGNOSTIC_TEST", "USABLE_SHORT_REFERENCE_DIAGNOSTIC")

# Nominal lookahead, but will dynamically adapt if the path is shorter.
NOMINAL_LOOKAHEAD_X_M = 0.45


class LaneStateFilter(Node):
    def __init__(self):
        super().__init__("lane_state_filter")

        self.declare_parameter("max_reference_age_s", 0.25)
        self.declare_parameter("max_missing_time_s", 0.50)
        self.declare_parameter("max_lateral_jump_m", 0.12)
        self.declare_parameter("max_heading_jump_rad", 0.45)
        self.declare_parameter("max_curvature_jump_inv_m", 3.0)
        self.declare_parameter("smoothing_alpha", 0.45)

        self.latest_result = self.unavailable("WAITING_FOR_REFERENCE")
        self.last_accepted = None
        self.last_accepted_monotonic = None
        self.last_source_stamp_ns = None
        self.last_message_monotonic = None

        self.create_subscription(
            String, REFERENCE_TOPIC, self.on_reference, 1
        )
        self.state_pub = self.create_publisher(
            String, STATE_TOPIC, 10
        )
        self.create_timer(0.1, self.publish_state)

        self.get_logger().info(
            "Camera-only lane-state diagnostic (Adaptive Lookahead); no motion commands."
        )

    @staticmethod
    def unavailable(reason, source_status=None, stamp_ns=None):
        result = {
            "valid": False,
            "reason": reason,
            "source_status": source_status,
            "lookahead_x_m": None,
            "center_y_left_m": None,
            "heading_rad": None,
            "curvature_inv_m": None,
            "confidence": 0.0,
            "reference_stamp_sec": None,
            "reference_stamp_nanosec": None,
        }

        if stamp_ns is not None:
            result["reference_stamp_sec"] = stamp_ns // 1_000_000_000
            result["reference_stamp_nanosec"] = stamp_ns % 1_000_000_000

        return result

    def reference_age_s(self, stamp_ns):
        now_ns = self.get_clock().now().nanoseconds
        return (now_ns - stamp_ns) / 1_000_000_000.0

    def on_reference(self, msg):
        arrival = time.monotonic()
        self.last_message_monotonic = arrival

        try:
            source = json.loads(msg.data)
            if not isinstance(source, dict):
                raise ValueError("Expected a JSON object")

            sec = int(source["stamp_sec"])
            nanosec = int(source["stamp_nanosec"])
            stamp_ns = sec * 1_000_000_000 + nanosec

        except (ValueError, TypeError, KeyError):
            self.latest_result = self.unavailable("MALFORMED_REFERENCE")
            return

        source_status = source.get("status")

        if stamp_ns == 0:
            self.latest_result = self.unavailable("MISSING_SOURCE_TIMESTAMP", source_status, stamp_ns)
            return

        if self.last_source_stamp_ns is not None and stamp_ns <= self.last_source_stamp_ns:
            self.latest_result = self.unavailable("NONINCREASING_SOURCE_TIMESTAMP", source_status, stamp_ns)
            return

        self.last_source_stamp_ns = stamp_ns

        age_s = self.reference_age_s(stamp_ns)
        max_age_s = float(self.get_parameter("max_reference_age_s").value)

        if age_s > max_age_s or age_s < -0.05:
            self.latest_result = self.unavailable("SOURCE_TIMESTAMP_NOT_FRESH", source_status, stamp_ns)
            return

        # Accept both full diagnostic routes and usable short routes
        if source_status not in VALID_STATUSES:
            self.latest_result = self.unavailable("REFERENCE_INVALID", source_status, stamp_ns)
            return

        try:
            points = np.asarray(source.get("points_x_forward_y_left_m", []), dtype=float)

            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5 or not np.isfinite(points).all():
                raise ValueError("Invalid metric point array")

            x = points[:, 0]
            y = points[:, 1]
            
            # Dynamic Lookahead: Do not extrapolate blindly into the void.
            # If the path ends before 0.45m, evaluate at the furthest reliable point instead.
            dynamic_lookahead_m = min(NOMINAL_LOOKAHEAD_X_M, float(x[-1]))

            # Fit y(x) = a*(x-x0)^2 + b*(x-x0) + c.
            centered_x = x - dynamic_lookahead_m
            design = np.column_stack((centered_x ** 2, centered_x, np.ones_like(x)))
            a, b, c = np.linalg.lstsq(design, y, rcond=None)[0]

            predicted_y = design @ np.array([a, b, c])
            residual_m = float(np.sqrt(np.mean((y - predicted_y) ** 2)))

            if residual_m > 0.08:
                raise ValueError("Reference shape is not locally well represented")

            measurement = {
                "center_y_left_m": float(c),
                "heading_rad": float(math.atan(b)),
                "curvature_inv_m": float(2.0 * a / ((1.0 + b * b) ** 1.5)),
            }

            if not all(math.isfinite(val) for val in measurement.values()):
                raise ValueError("Non-finite lane state")

        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as e:
            self.latest_result = self.unavailable(f"BAD_METRIC_POINTS: {e}", source_status, stamp_ns)
            return

        max_missing_s = float(self.get_parameter("max_missing_time_s").value)

        tracking_recent = (
            self.last_accepted is not None
            and self.last_accepted_monotonic is not None
            and arrival - self.last_accepted_monotonic <= max_missing_s
        )

        if tracking_recent:
            limits = {
                "center_y_left_m": float(self.get_parameter("max_lateral_jump_m").value),
                "heading_rad": float(self.get_parameter("max_heading_jump_rad").value),
                "curvature_inv_m": float(self.get_parameter("max_curvature_jump_inv_m").value),
            }

            if any(abs(measurement[key] - self.last_accepted[key]) > limit for key, limit in limits.items()):
                self.latest_result = self.unavailable("MEASUREMENT_JUMP", source_status, stamp_ns)
                return

            alpha = float(self.get_parameter("smoothing_alpha").value)
            alpha = min(1.0, max(0.0, alpha))

            estimate = {
                key: (alpha * measurement[key] + (1.0 - alpha) * self.last_accepted[key])
                for key in measurement
            }
        else:
            estimate = measurement

        self.last_accepted = estimate
        self.last_accepted_monotonic = arrival

        pairs = max(0, int(source.get("pairs", 0)))
        singles = max(0, int(source.get("single", 0)))
        pair_fraction = pairs / max(1, pairs + singles)

        # Dynamic confidence scoring
        coverage_score = min(1.0, float(source.get("observed_span_m", 0.0)) / 0.65)
        fit_score = max(0.0, 1.0 - residual_m / 0.06)
        
        # Penalize confidence if the lookahead had to be shortened (sharp curve penalty)
        lookahead_penalty = max(0.0, (NOMINAL_LOOKAHEAD_X_M - dynamic_lookahead_m) * 1.5)

        confidence = (0.25 + 0.30 * coverage_score + 0.25 * fit_score + 0.20 * pair_fraction) - lookahead_penalty

        self.latest_result = {
            "valid": True,
            "reason": "CAMERA_MEASUREMENT_ACCEPTED",
            "source_status": source_status,
            "lookahead_x_m": round(dynamic_lookahead_m, 4),
            "center_y_left_m": round(estimate["center_y_left_m"], 4),
            "heading_rad": round(estimate["heading_rad"], 4),
            "curvature_inv_m": round(estimate["curvature_inv_m"], 4),
            "confidence": round(min(1.0, max(0.0, confidence)), 3),
            "confidence_is_probability": False,
            "fit_residual_m": round(residual_m, 4),
            "reference_near_x_m": round(float(x[0]), 4),
            "reference_far_x_m": round(float(x[-1]), 4),
            "reference_stamp_sec": sec,
            "reference_stamp_nanosec": nanosec,
        }

    def publish_state(self):
        result = self.latest_result.copy()
        now = time.monotonic()

        if result["valid"]:
            stamp_ns = result["reference_stamp_sec"] * 1_000_000_000 + result["reference_stamp_nanosec"]

            if (
                self.last_message_monotonic is None
                or now - self.last_message_monotonic > float(self.get_parameter("max_reference_age_s").value)
                or self.reference_age_s(stamp_ns) > float(self.get_parameter("max_reference_age_s").value)
            ):
                self.latest_result = self.unavailable("REFERENCE_STALE", result["source_status"], stamp_ns)
                result = self.latest_result.copy()

        output = String()
        output.data = json.dumps(result)
        self.state_pub.publish(output)

        self.get_logger().info(
            f"VALID {result['valid']} "
            f"REASON {result['reason']} "
            f"LOOKAHEAD {result['lookahead_x_m']} m "
            f"CENTER_Y {result['center_y_left_m']} m "
            f"HEADING {result['heading_rad']} rad "
            f"CURVATURE {result['curvature_inv_m']} 1/m "
            f"CONFIDENCE {result['confidence']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaneStateFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()