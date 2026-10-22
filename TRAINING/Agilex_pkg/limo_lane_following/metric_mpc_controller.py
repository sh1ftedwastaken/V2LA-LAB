#!/usr/bin/env python3
"""Experimental motion MPC for metric lane references.

Input:  /lane/metric_reference_debug  std_msgs/String JSON
Output: /cmd_vel_request             geometry_msgs/Twist
        /lane/mpc_motion_debug       std_msgs/String JSON

Requires a separately enabled, fail-closed safety supervisor.
"""

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from scipy.optimize import minimize
from std_msgs.msg import String


FULL = "VALID_FOR_DIAGNOSTIC_TEST"
SHORT = "USABLE_SHORT_REFERENCE_DIAGNOSTIC"


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def prepare_path(source):
    """Check and resample the ordered path; never sort it by X."""
    if source.get("status") not in (FULL, SHORT):
        raise ValueError("REFERENCE_INVALID")

    points = np.asarray(
        source["points_x_forward_y_left_m"], dtype=float
    )
    if (
        points.ndim != 2 or points.shape[1] != 2
        or len(points) < 5 or not np.isfinite(points).all()
    ):
        raise ValueError("BAD_POINTS")

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(steps < 0.0005) or np.any(steps > 0.25):
        raise ValueError("DISCONNECTED_OR_DUPLICATE_PATH")

    arc = float(np.sum(steps))
    if arc < 0.14:
        raise ValueError("INSUFFICIENT_ARC")

    first = points[0]
    if (
        first[0] < 0.08 or first[0] > 0.45
        or abs(first[1]) > 0.18
    ):
        raise ValueError("UNSUPPORTED_NEAR_GAP")

    # Only an initial connection across the camera blind spot;
    # it is NOT a camera observation.
    path = np.vstack(([0.0, 0.0], points))
    segment = np.diff(path, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths < 0.001):
        raise ValueError("DEGENERATE_PATH")

    headings = np.unwrap(
        np.arctan2(segment[:, 1], segment[:, 0])
    )
    # This version models forward driving, not reversing or
    # turning around on the spot.
    # if np.any(np.abs(headings) > 1.45):
    #     raise ValueError("FORWARD_MODEL_UNSUPPORTED")

    s = np.r_[0.0, np.cumsum(lengths)]
    return path, s, headings, arc


class MetricMPC(Node):
    def __init__(self):
        super().__init__("metric_mpc_controller")

        self.request_pub = self.create_publisher(
            Twist, "/cmd_vel_request", 10
        )
        self.debug_pub = self.create_publisher(
            String, "/lane/mpc_motion_debug", 10
        )
        self.create_subscription(
            String, "/lane/metric_reference_debug",
            self.on_reference, 1
        )

        self.latest = None
        self.last_stamp_ns = None
        self.last_arrival = None
        self.previous_command = np.zeros(2)
        self.last_log_key = None
        self.last_log_time = 0.0

        self.dt = 0.10
        self.n = 8
        self.max_v = 0.25
        self.max_w = 0.45
        self.max_dv = 0.03     # m/s per 0.10 s
        self.max_dw = 0.08      # rad/s per 0.10 s
        self.max_frame_age = 0.22
        self.max_solve_time = 0.15

        self.create_timer(self.dt, self.tick)
        self.get_logger().warn(
            "EXPERIMENTAL MOTION MPC. Start supervisor DISARMED."
        )

    def on_reference(self, msg):
        try:
            source = json.loads(msg.data)
            if not isinstance(source, dict):
                raise ValueError("not an object")
            sec = int(source["stamp_sec"])
            nsec = int(source["stamp_nanosec"])
            if sec < 0 or not 0 <= nsec < 1_000_000_000:
                raise ValueError("bad stamp")
            stamp = sec * 1_000_000_000 + nsec
            if stamp == 0 or (
                self.last_stamp_ns is not None
                and stamp <= self.last_stamp_ns
            ):
                raise ValueError("nonincreasing stamp")
        except (KeyError, TypeError, ValueError) as exc:
            self.latest = None
            self.report("BAD_REFERENCE_MESSAGE", str(exc))
            return

        self.last_stamp_ns = stamp
        self.last_arrival = time.monotonic()
        self.latest = source

    def report(self, status, detail=None, v=0.0, omega=0.0):
        payload = {
            "status": status,
            "detail": detail,
            "requested_v_mps": round(v, 4),
            "requested_omega_rps": round(omega, 4),
            "motion_request": status == "MPC_REQUEST",
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.debug_pub.publish(msg)

        now = time.monotonic()
        key = (status, detail)
        if key != self.last_log_key or now - self.last_log_time >= 1.0:
            line = (
                f"{status} detail={detail} "
                f"v={v:.4f} m/s omega={omega:.4f} rad/s"
            )
            if status == "MPC_REQUEST":
                self.get_logger().info(line)
            else:
                self.get_logger().warn(line)
            self.last_log_key = key
            self.last_log_time = now

    def stop(self, reason):
        self.previous_command = np.zeros(2)
        self.request_pub.publish(Twist())
        self.report("STOP", reason)

    def tick(self):
        if self.latest is None or self.last_arrival is None:
            self.stop("NO_REFERENCE")
            return

        now = time.monotonic()
        source = self.latest
        stamp = self.last_stamp_ns
        age = (
            self.get_clock().now().nanoseconds - stamp
        ) / 1e9

        if (
            now - self.last_arrival > self.max_frame_age
            or age < -0.05
            or age > self.max_frame_age
        ):
            self.stop("STALE_REFERENCE")
            return

        try:
            path, s, headings, observed_arc = prepare_path(source)
        except (ValueError, KeyError, TypeError) as exc:
            self.stop(str(exc))
            return

        # Reduce target speed on a short reference. The preview
        # never extends past the last supplied reference point.
        target_v = (
            0.04 if source["status"] == SHORT else self.max_v
        )
        target_v = min(target_v, self.max_v)
        if s[-1] < 0.35:
            target_v = min(target_v, 0.025)

        def target_at(distance):
            d = min(float(distance), float(s[-1]))
            x = float(np.interp(d, s, path[:, 0]))
            y = float(np.interp(d, s, path[:, 1]))
            j = int(np.searchsorted(s[1:], d, side="left"))
            j = min(j, len(headings) - 1)
            return x, y, float(headings[j])

        n = self.n
        dt = self.dt
        last_v, last_w = self.previous_command

        def rollout(flat):
            commands = flat.reshape(n, 2)
            x = y = theta = 0.0
            cost = 0.0
            prev_v, prev_w = last_v, last_w

            for k, (v, w) in enumerate(commands):
                # Midpoint integration of differential-drive motion.
                x += dt * v * math.cos(theta + 0.5 * dt * w)
                y += dt * v * math.sin(theta + 0.5 * dt * w)
                theta = wrap(theta + dt * w)

                rx, ry, rh = target_at(
                    (k + 1) * target_v * dt
                )
                cost += (
                    65.0 * (y - ry) ** 2
                    + 8.0 * (x - rx) ** 2
                    + 1.5 * wrap(theta - rh) ** 2
                    + 0.6 * (v - target_v) ** 2
                    + 0.10 * w ** 2
                    + 0.8 * (w - prev_w) ** 2
                    + 2.0 * (v - prev_v) ** 2
                )
                prev_v, prev_w = v, w

            return cost

        def rate_limits(flat):
            commands = flat.reshape(n, 2)
            previous = np.vstack(
                (self.previous_command, commands[:-1])
            )
            changes = commands - previous
            return np.column_stack((
                self.max_dv - changes[:, 0],
                self.max_dv + changes[:, 0],
                self.max_dw - changes[:, 1],
                self.max_dw + changes[:, 1],
            )).ravel()

        initial = np.tile(
            [min(target_v, last_v + self.max_dv), last_w], n
        )
        bounds = [
            (0.0, target_v) if i % 2 == 0
            else (-self.max_w, self.max_w)
            for i in range(2 * n)
        ]

        started = time.monotonic()
        try:
            solution = minimize(
                rollout, initial, method="SLSQP",
                bounds=bounds,
                constraints=[{
                    "type": "ineq",
                    "fun": rate_limits,
                }],
                options={"maxiter": 45, "ftol": 1e-5,
                         "disp": False},
            )
        except Exception as exc:
            self.stop(f"SOLVER_EXCEPTION: {exc}")
            return

        solve_time = time.monotonic() - started
        if (
            not solution.success
            or solution.x is None
            or not np.isfinite(solution.x).all()
            or solve_time > self.max_solve_time
            or np.min(rate_limits(solution.x)) < -1e-4
        ):
            self.stop(
                f"SOLVER_NOT_READY: {solution.message}; "
                f"solve_s={solve_time:.3f}"
            )
            return

        v, w = map(float, solution.x[:2])
        if v < -1e-5 or abs(w) > self.max_w + 1e-4:
            self.stop("OUTPUT_OUT_OF_BOUNDS")
            return

        cmd = Twist()
        cmd.linear.x = max(0.0, min(self.max_v, v))
        cmd.angular.z = max(-self.max_w, min(self.max_w, w))
        self.previous_command[:] = (cmd.linear.x, cmd.angular.z)
        self.request_pub.publish(cmd)
        self.report(
            "MPC_REQUEST", f"arc={observed_arc:.3f}m; "
            f"solve={solve_time:.3f}s; "
            f"source={source['status']}",
            cmd.linear.x, cmd.angular.z
        )


def main(args=None):
    rclpy.init(args=args)
    node = MetricMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.request_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()