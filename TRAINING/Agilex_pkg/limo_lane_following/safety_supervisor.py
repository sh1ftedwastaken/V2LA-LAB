#!/usr/bin/env python3
"""Forward MPC requests to the LIMO; stop for front LiDAR or lost input."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class SafetySupervisor(Node):
    def __init__(self):
        super().__init__("safety_supervisor")

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Twist, "/cmd_vel_request", self.on_request, 10
        )
        self.create_subscription(
            LaserScan, "/scan", self.lidar_cb, qos_profile_sensor_data
        )

        self.request = Twist()
        self.request_time = None
        self.scan_time = None
        self.front_clear = False
        self.last_reason = None

        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            "Forwarding /cmd_vel_request when front LiDAR is clear."
        )

    def on_request(self, msg):
        if not all(math.isfinite(x) for x in
                   (msg.linear.x, msg.angular.z)):
            self.request_time = None
            return
        self.request = msg
        self.request_time = time.monotonic()

    def lidar_cb(self, msg):
        self.scan_time = time.monotonic()
        front = []

        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) > 0.26:
                continue

            if math.isnan(distance):
                continue

            if math.isinf(distance):
                # Common LaserScan convention: no object returned
                # within the scanner's measurable range.
                continue

            if msg.range_min <= distance <= msg.range_max:
                front.append(distance)

        # Require at least one measurable front reading. Do not
        # interpret an absent or entirely unreadable scan as clear.
        self.front_clear = bool(front) and min(front) >= 0.35

    def tick(self):
        now = time.monotonic()
        output = Twist()

        if self.scan_time is None or now - self.scan_time > 0.25:
            reason = "NO_FRESH_SCAN"
        elif not self.front_clear:
            reason = "FRONT_NOT_CLEAR"
        elif self.request_time is None or now - self.request_time > 0.20:
            reason = "NO_FRESH_MPC_REQUEST"
        else:
            reason = "FORWARDING_MPC"
            output.linear.x = max(
                0.0, min(0.25, self.request.linear.x)
            )
            output.angular.z = max(
                -0.43, min(0.43, self.request.angular.z)
            )

        self.pub.publish(output)

        if reason != self.last_reason:
            self.get_logger().info(
                f"{reason}: v={output.linear.x:.3f} "
                f"omega={output.angular.z:.3f}"
            )
            self.last_reason = reason


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()