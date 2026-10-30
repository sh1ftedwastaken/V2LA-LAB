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
    """Check, filter duplicates, and resample the ordered path."""
    if source.get("status") not in (FULL, SHORT):
        raise ValueError("REFERENCE_INVALID")

    raw_points = np.asarray(
        source["points_x_forward_y_left_m"], dtype=float
    )
    if (
        raw_points.ndim != 2 or raw_points.shape[1] != 2
        or len(raw_points) < 5 or not np.isfinite(raw_points).all()
    ):
        raise ValueError("BAD_POINTS")

    # Filter out duplicate/ultra-close points instead of crashing
    clean_points = [raw_points[0]]
    for i in range(1, len(raw_points)):
        if np.linalg.norm(raw_points[i] - clean_points[-1]) >= 0.001:  # 1 mm minimum step
            clean_points.append(raw_points[i])
            
    points = np.asarray(clean_points)
    if len(points) < 5:
        raise ValueError("BAD_POINTS_AFTER_FILTERING")

    # Allowed step grows with distance (perspective), matching the
    # camera reference. Keep the connected prefix; drop the far tail.
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    limits = np.maximum(0.15, 0.20 * points[1:, 0])
    bad = np.flatnonzero(steps > limits)
    if len(bad):
        points = points[: bad[0] + 1]   # truncate at first real jump
        steps = steps[: bad[0]]

    # Optional: ignore points far beyond the calibrated range (0.93 m)
    keep = points[:, 0] <= 1.2
    if keep.sum() >= 5:
        points = points[keep]
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)

    if len(points) < 5:
        raise ValueError("DISCONNECTED_PATH_TOO_SHORT")
    
    window = min(10, len(points))
    if window % 2 == 0:
        window -= 1 
        
    if window >= 3:
        w = np.ones(window) / window
        pad = window // 2
        px = np.pad(points[:, 0], pad, mode='edge')
        py = np.pad(points[:, 1], pad, mode='edge')
        points[:, 0] = np.convolve(px, w, mode='valid')
        points[:, 1] = np.convolve(py, w, mode='valid')
        
        # Recompute steps after smoothing
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)

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

    s = np.r_[0.0, np.cumsum(lengths)]
    return path, s, headings, arc


def curvature_speed(path, s, v_max, a_lat, max_w, window=0.6):
    """Estimate curvature from local quadratic fits to X(s), Y(s).

    Excludes the artificial bumper-to-first-reference connection.
    Does not extrapolate past the supplied reference.
    """
    observed_s = np.asarray(s[1:], dtype=float)
    observed_points = np.asarray(path[1:], dtype=float)

    if len(observed_points) < 5:
        raise ValueError("CURVATURE_INSUFFICIENT_POINTS")

    start = float(observed_s[0])
    end = min(float(observed_s[-1]), start + window)

    if end - start < 0.14:
        raise ValueError("CURVATURE_INSUFFICIENT_SPAN")

    # Uniform metric sampling prevents dense near-image points
    # from dominating the fit.
    count = max(8, int(math.ceil((end - start) / 0.025)) + 1)
    grid = np.linspace(start, end, count)
    xy = np.column_stack((
        np.interp(grid, observed_s, observed_points[:, 0]),
        np.interp(grid, observed_s, observed_points[:, 1]),
    ))

    estimates = []
    residuals = []
    locations = []

    for center in grid:
        # Approximately 0.25 m support, shifted inward at endpoints.
        lo = max(start, min(float(center) - 0.125, end - 0.25))
        hi = min(end, lo + 0.25)
        selected = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)

        local_s = grid[selected] - center
        local_xy = xy[selected]

        if len(local_s) < 5:
            continue

        design = np.column_stack((
            np.ones_like(local_s),
            local_s,
            local_s ** 2,
        ))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, local_xy, rcond=None
        )
        if rank < 3:
            continue

        fitted = design @ coefficients
        residual = float(np.sqrt(np.mean(
            np.sum((fitted - local_xy) ** 2, axis=1)
        )))

        # A substantial mismatch is not something to smooth away.
        if residual > 0.080:
            raise ValueError(
                f"CURVATURE_FIT_CONFLICT:"
                f"residual={residual:.3f}m;"
                f"s_from_first={center - start:.3f}m"
            )
        
        dx, dy = coefficients[1]
        ddx, ddy = 2.0 * coefficients[2]
        tangent_norm = math.hypot(float(dx), float(dy))

        safe_tangent = max(tangent_norm, 0.30)
        
        curvature = float(
            (dx * ddy - dy * ddx) / safe_tangent ** 3
        )
        if not math.isfinite(curvature):
            raise ValueError("CURVATURE_NONFINITE")

        estimates.append(abs(curvature))
        residuals.append(residual)
        locations.append(float(center - start))

    if not estimates:
        raise ValueError("CURVATURE_NO_USABLE_FIT")

    peak = int(np.argmax(estimates))
    kappa = estimates[peak]

    limit = v_max
    if kappa >= 1e-3:
        limit = min(
            v_max,
            math.sqrt(a_lat / kappa),
            max_w / kappa,
        )

    diagnostics = {
        "peak_s_from_first_m": locations[peak],
        "max_fit_residual_m": max(residuals),
    }
    return float(limit), float(kappa), diagnostics


class MetricMPC(Node):
    def __init__(self):
        super().__init__("metric_mpc_controller")
        
        self.declare_parameter("max_v", 0.23)

        self.request_pub = self.create_publisher(
            Twist, "/cmd_vel_request", 10
        )
        self.debug_pub = self.create_publisher(
            String, "/lane/mpc_motion_debug", 10
        )
        self.create_subscription(
            String, "/lane/metric_reference",
            self.on_reference, 1
        )
        
        self.latest           = None
        self.last_stamp_ns    = None
        self.last_arrival     = None
        self.previous_command = np.zeros(2)
        self.last_log_key     = None
        self.last_log_time    = 0.0

        self.last_good_command_time = None
        self.loss_started = None
        self.loss_initial_command = np.zeros(2)
        self.max_reference_loss_s = 0.70
        
        self.a_lat = 0.35
        self.dt = 0.10
        self.n  = 8
        self.max_v   = float(self.get_parameter("max_v").value)
        self.max_w   = 0.43
        self.max_dv  = 0.05      
        self.max_dw  = 0.07
        self.max_frame_age  = 0.45
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
            
    def slow_for_missing_reference(self, reason):
        now = time.monotonic()

        if self.last_good_command_time is None:
            self.stop(reason)
            return

        if self.loss_started is None:
            self.loss_started = now
            self.loss_initial_command = self.previous_command.copy()

        elapsed = now - self.loss_started

        if elapsed >= self.max_reference_loss_s:
            self.stop(reason)
            return

        # Begin reducing on the first failed tick. Speed and turning
        # fade together from the command at the start of this loss.
        fraction = max(
            0.0,
            1.0 - (elapsed + self.dt) / self.max_reference_loss_s,
        )

        v = float(self.loss_initial_command[0] * fraction)
        w = float(self.loss_initial_command[1] * fraction)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w

        self.previous_command[:] = (v, w)
        self.request_pub.publish(cmd)
        self.report("DECELERATING", reason, v, w)

    def stop(self, reason):
        self.previous_command = np.zeros(2)
        self.request_pub.publish(Twist())
        self.report("STOP", reason)

    def tick(self):
        if self.latest is None or self.last_arrival is None:
            self.slow_for_missing_reference("NO_REFERENCE")
            return

        now = time.monotonic()
        source = self.latest
        stamp = self.last_stamp_ns
        age = (
            self.get_clock().now().nanoseconds - stamp
        ) / 1e9

        arrival_age = now - self.last_arrival
        if (
            arrival_age > self.max_frame_age
            or age < -0.05
            or age > self.max_frame_age
        ):
            self.slow_for_missing_reference(
                f"STALE_REFERENCE:"
                f"source_age={age:.3f}s;"
                f"arrival_age={arrival_age:.3f}s;"
                f"limit={self.max_frame_age:.3f}s"
            )
            return

        try:
            path, s, headings, observed_arc = prepare_path(source)

            curve_limit, kappa, curvature_debug = curvature_speed(
                path, s, self.max_v, self.a_lat, self.max_w
            )
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            self.slow_for_missing_reference(str(exc))
            return

        coverage_limit = max(
            0.0,
            (observed_arc - 0.06) / (self.n * self.dt),
        )

        target_v = min(
            self.max_v, curve_limit, coverage_limit
        )
        
        coverage_limit = max(
            0.0, (observed_arc - 0.06) / (self.n * self.dt)
        )
        
        target_v = min(self.max_v, curve_limit, coverage_limit)

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

        # Speed cap
        speed_cap = min(
            self.max_v,
            max(target_v, float(last_v) - self.max_dv)
        )
        initial_v = float(np.clip(
            target_v,
            max(0.0, float(last_v) - self.max_dv),
            min(speed_cap, float(last_v) + self.max_dv),
        ))

        initial = np.tile([initial_v, last_w], n)
        bounds = [
            (0.0, speed_cap) if i % 2 == 0
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
            self.slow_for_missing_reference(f"SOLVER_EXCEPTION: {exc}")
            return

        solve_time = time.monotonic() - started
        if (
            not solution.success
            or solution.x is None
            or not np.isfinite(solution.x).all()
            or solve_time > self.max_solve_time
            or np.min(rate_limits(solution.x)) < -1e-4
        ):
            self.slow_for_missing_reference(
                f"SOLVER_NOT_READY: {solution.message}; "
                f"solve_s={solve_time:.3f}"
            )
            return

        v, w = map(float, solution.x[:2])
        if v < -1e-5 or abs(w) > self.max_w + 1e-4:
            self.slow_for_missing_reference("OUTPUT_OUT_OF_BOUNDS")
            return

        cmd = Twist()
        cmd.linear.x = float(np.clip(v, 0.0, self.max_v))
        cmd.angular.z = float(np.clip(w, -self.max_w, self.max_w))
        self.previous_command[:] = (
            cmd.linear.x, cmd.angular.z
        )
        self.last_good_command_time = time.monotonic()
        self.request_pub.publish(cmd)
        self.report(
            "MPC_REQUEST",
            f"arc={observed_arc:.3f}m; "
            f"kappa={kappa:.2f}; "
            f"peak_s={curvature_debug['peak_s_from_first_m']:.3f}m; "
            f"fit_err={curvature_debug['max_fit_residual_m']:.4f}m; "
            f"curve_limit={curve_limit:.3f}; "
            f"coverage_limit={coverage_limit:.3f}; "
            f"target_v={target_v:.3f}; "
            f"source_age_start={age:.3f}s; "
            f"solve={solve_time:.3f}s; "
            f"source={source['status']}",
            cmd.linear.x,
            cmd.angular.z,
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