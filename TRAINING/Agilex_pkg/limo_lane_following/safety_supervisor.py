#!/usr/bin/env python3
"""
safety_supervisor.py
====================
Layer 5: Safety supervisor
Ensures the robot stops if the controller crashes or an obstacle is detected.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

class SafetySupervisor(Node):
    def __init__(self):
        super().__init__('safety_supervisor')
        
        # Publisher for the actual motor commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriptions
        self.req_sub = self.create_subscription(Twist, '/cmd_vel_request', self.cmd_req_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, rclpy.qos.qos_profile_sensor_data)
        
        # Watchdog timer runs at 10Hz
        self.timer = self.create_timer(0.1, self.watchdog_cb)
        
        self.last_cmd_time = time.time()
        self.current_cmd = Twist()
        
        # Safety states
        self.obstacle_detected = False
        self.safety_stop_distance = 0.30  # meters
        
        self.get_logger().info("Layer 5: Safety Supervisor Initialized.")
        self.get_logger().info("Listening to /cmd_vel_request and publishing to /cmd_vel")

    def lidar_cb(self, msg: LaserScan):
        """Run the LiDAR stop in its own callback."""
        front_ranges = []
        for index, distance in enumerate(msg.ranges):
            angle = msg.angle_min + index * msg.angle_increment
            if not (msg.range_min < distance < msg.range_max):
                continue
            
            # Check a narrow cone directly in front of the robot (-15 to 15 degrees)
            if -0.26 <= angle <= 0.26:
                front_ranges.append(distance)
                
        min_dist = min(front_ranges) if front_ranges else float('inf')
        
        if min_dist < self.safety_stop_distance:
            if not self.obstacle_detected:
                self.get_logger().warn(f"STOP: Obstacle detected at {min_dist:.2f}m!")
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    def cmd_req_cb(self, msg: Twist):
        """Receive requested commands from the Layer 4 controller."""
        self.last_cmd_time = time.time()
        self.current_cmd = msg

    def watchdog_cb(self):
        """
        Send zero if the controller command is older than 0.2 s, 
        or if an obstacle is detected.
        """
        time_since_last_cmd = time.time() - self.last_cmd_time
        
        safe_cmd = Twist()
        
        # Rule 1: Watchdog Timeout (Command older than 0.2 seconds)
        # Rule 2: LiDAR Emergency Stop
        if time_since_last_cmd > 0.2 or self.obstacle_detected:
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = 0.0
            
        # Rule 3: Normal Operation
        else:
            safe_cmd = self.current_cmd
            
        # Only the supervisor publishes to the actual motors
        self.cmd_pub.publish(safe_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()