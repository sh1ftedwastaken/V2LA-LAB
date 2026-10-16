#!/usr/bin/env python3
"""
metric_ground_frame.py
======================
Layer 2: Metric Conversion
Converts pixel coordinates from the camera into physical meters relative 
to the robot's front bumper (X forward, Y left).
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

class MetricGroundFrame(Node):
    def __init__(self):
        super().__init__('metric_ground_frame')

        # 1. Source points in the RAW CAMERA IMAGE (Pixels)
        self.src_pixels = np.float32([
            [11.0, 238.0],    # Bottom-Left
            [305.5, 238.0],   # Bottom-Right
            [210.0, 160.0],   # Top-Right
            [105.0, 160.0],   # Top-Left
        ])

        # 2. Destination points in the REAL WORLD (Meters)
        # Coordinates are (X forward, Y left) relative to the bumper at (0,0).
        self.dst_meters = np.float32([
            [0.305, +0.210],    # bottom-left: forward, left
            [0.305, -0.210],    # bottom-right: forward, right
            [0.930, -0.2115],   # top-right: forward, right
            [0.930, +0.2115],   # top-left: forward, left
        ])

        # 3. Calculate the Pixel-to-Meters Homography Matrix
        self.H_pixel_to_meter = cv2.getPerspectiveTransform(self.src_pixels, self.dst_meters)

        self.get_logger().info("Layer 2: Metric Ground Frame Initialized.")
        self.get_logger().info("Ready to convert pixels to physical meters.")

    def convert_pixel_path_to_meters(self, pixel_path: np.ndarray) -> np.ndarray:
        """
        Takes an Nx2 array of (x, y) pixel coordinates and returns 
        an Nx2 array of (X, Y) meter coordinates.
        """
        if pixel_path is None or len(pixel_path) == 0:
            return np.array([])

        # OpenCV perspectiveTransform requires a specific 3D shape: (N, 1, 2)
        pts_px = np.array(pixel_path, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply the matrix to convert pixels to meters
        pts_m = cv2.perspectiveTransform(pts_px, self.H_pixel_to_meter)
        
        # Reshape back to Nx2
        return pts_m.reshape(-1, 2)

def main(args=None):
    rclpy.init(args=args)
    node = MetricGroundFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()