"""
seg_bev_node.py
---------------
ROS2 node: subscribes to a segmentation mask,
warps it to BEV, publishes cleaned BEV mask.  

Topics subscribed:
    /seg/mask_raw               (sensor_msgs/Image)  - raw seg mask (mono8)  

Topics published:
    /seg/bev_mask               (sensor_msgs/Image)  - BEV cleaned mask (mono8)
    /seg/bev                    (sensor_msgs/Image)  - colourised BEV for debugging
    /seg/bev_overlay            (sensor_msgs/Image)  - driving ROI and center debug overlay  
    /seg/bev_grid               (sensor_msgs/Image)  - calibration trapezoid overlay (when calibrate_mode:=True)
    
Run:
    ros2 run limo_seg seg_bev_node --ros-args -p calibrate_mode:=True
"""
from __future__ import annotations
import os
import sys  
import cv2
import numpy as np
import yaml
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bev.bev_transform import undistort_mask, to_bev, cleanup_bev, scale_intrinsics

# Import shared lane analysis from lane_analyzer.py
from lane_analyzer import (
    LaneAnalyzer,
    LaneOverlayRenderer,
    CLASS_BG,
    CLASS_ROAD,
    CLASS_WHITE,
    CLASS_YELLOW,
    CLASS_VEHICLE,
    CURVE_YELLOW,
    CURVE_WHITE,
    CURVE_CENTER,
    OBSTACLE_BOX,
    EGO_AXIS,
    STATE_BOTH,
    STATE_LEFT_ONLY,
    STATE_RIGHT_ONLY,
    STATE_LOST,
    DEFAULT_LANE_WIDTH_PX,
    DEFAULT_ROI_START_RATIO,
    DEFAULT_ROI_END_RATIO
)  

class SegBEVNode(Node):  
    def __init__(self):
        super().__init__("seg_bev_node")  
        
        self.declare_parameter("mask_topic", "/seg/mask_raw")
        self.declare_parameter("camera_params", "bev2/camera_params.txt")
        self.declare_parameter("bev_size", 160)
        self.declare_parameter("mask_width", 160)
        self.declare_parameter("mask_height", 120)
        self.declare_parameter("default_lane_width", DEFAULT_LANE_WIDTH_PX)
        self.declare_parameter("camera_offset_x_px", 0.0)
        self.declare_parameter("roi_start_ratio", DEFAULT_ROI_START_RATIO)
        self.declare_parameter("roi_end_ratio", DEFAULT_ROI_END_RATIO)
        self.declare_parameter("calibrate_mode", False)
        self.declare_parameter("debug_mode", False)

        
        # BEV Source Parameters
        self.declare_parameter("bev_src_bottom_left_x", 10)       # 15
        self.declare_parameter("bev_src_bottom_left_y", 120.0)      # 240   
        self.declare_parameter("bev_src_bottom_right_x", 155.5)     # 315
        self.declare_parameter("bev_src_bottom_right_y", 120.0)     # 240
        self.declare_parameter("bev_src_top_right_x", 108)        # 215
        self.declare_parameter("bev_src_top_right_y", 80.0)        # 160
        self.declare_parameter("bev_src_top_left_x", 56.0)         # 100
        self.declare_parameter("bev_src_top_left_y", 80.0)         # 160
        
        # BEV Destination Parameters (Margin-based)
        self.declare_parameter("bev_dst_margin_x", 20.0)
        self.declare_parameter("bev_dst_top_y", 5.0)
        self.declare_parameter("bev_dst_bottom_y", 162)
        
        # Parameters for lane analysis and overlay
        self.bev_size           = self.get_parameter("bev_size").value
        self.mask_w             = self.get_parameter("mask_width").value
        self.mask_h             = self.get_parameter("mask_height").value
        mask_topic              = self.get_parameter("mask_topic").value
        params_path             = self.get_parameter("camera_params").value
        self.camera_offset_x_px = self.get_parameter("camera_offset_x_px").value
        self.lane_width_px      = self.get_parameter("default_lane_width").value
        self.roi_start_ratio    = self.get_parameter("roi_start_ratio").value
        self.roi_end_ratio      = self.get_parameter("roi_end_ratio").value
        self.debug_mode = self.get_parameter("debug_mode").value
        
        with open(params_path, "r", encoding="utf-8") as handle:
            cam = yaml.safe_load(handle)  
            
        k_native = np.array(cam["k"]).reshape(3, 3)
        self.d = np.array(cam["d"])
        self.k = scale_intrinsics(
            k_native,
            native_w=cam["width"],
            native_h=cam["height"],
            target_w=self.mask_w,
            target_h=self.mask_h,
        )  
        
        # Initialize shared lane analyzer and renderer
        self.analyzer = LaneAnalyzer(
            lane_width_px=self.lane_width_px,
            camera_offset_x_px=self.camera_offset_x_px,
            roi_start_ratio=self.roi_start_ratio,
            roi_end_ratio=self.roi_end_ratio,
            alpha_lane_width=0.05,
        )
        self.renderer = LaneOverlayRenderer()  
        self.bridge = CvBridge()
        self._frames = 0  
        
        self.sub          = self.create_subscription(Image, mask_topic, self.image_callback, 10)
        self.pub_bev_mask = self.create_publisher(Image, "/seg/bev_mask", 10)
        self.pub_bev      = self.create_publisher(Image, "/seg/bev", 10)
        self.pub_overlay  = self.create_publisher(Image, "/seg/bev_overlay", 10)  
        self.pub_bev_grid = self.create_publisher(Image, "/seg/bev_grid", 10)
        self.pub_bev_debug = self.create_publisher(Image, "/seg/bev_debug", 10)
        
        self.get_logger().info(f"Subscribed to mask topic: {mask_topic}")
        self.get_logger().info(f"Expected mask size     : {self.mask_w}x{self.mask_h}")
        self.get_logger().info("seg_bev_node ready - waiting for masks (supports live calibration)")  

    def _get_src_points(self) -> np.ndarray:
        """Fetches live ROS parameters for source points."""
        return np.float32([
            [self.get_parameter("bev_src_bottom_left_x").value,  self.get_parameter("bev_src_bottom_left_y").value],
            [self.get_parameter("bev_src_bottom_right_x").value, self.get_parameter("bev_src_bottom_right_y").value],
            [self.get_parameter("bev_src_top_right_x").value,    self.get_parameter("bev_src_top_right_y").value],
            [self.get_parameter("bev_src_top_left_x").value,     self.get_parameter("bev_src_top_left_y").value],
        ])
        
    def _get_dst_points(self) -> np.ndarray:
        """Fetches live ROS parameters for destination points."""
        return np.float32([
            [self.get_parameter("bev_dst_margin_x").value,                 self.get_parameter("bev_dst_bottom_y").value],
            [self.bev_size - self.get_parameter("bev_dst_margin_x").value, self.get_parameter("bev_dst_bottom_y").value],
            [self.bev_size - self.get_parameter("bev_dst_margin_x").value, self.get_parameter("bev_dst_top_y").value],
            [self.get_parameter("bev_dst_margin_x").value,                 self.get_parameter("bev_dst_top_y").value],
        ])

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls, bgr in {
            CLASS_BG: (0, 255, 0),
            CLASS_ROAD: (100, 100, 100),
            CLASS_WHITE: (255, 255, 255),
            CLASS_YELLOW: (0, 255, 255),
            CLASS_VEHICLE: (0, 0, 255),
        }.items():
            color[mask == cls] = bgr
        return color  
        
    def _draw_calib(self, mask: np.ndarray, src_points: np.ndarray) -> np.ndarray:
        """Draws the calibration trapezoid over the mask."""
        vis = self.colorize(mask)
        pts = src_points.astype(np.int32).reshape(-1, 1, 2)
        
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 255), thickness=1)
        
        labels = ["0:BL", "1:BR", "2:TR", "3:TL"]
        for (x, y), lbl in zip(src_points.astype(int), labels):
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(vis, lbl, (x + 4, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        return vis  

    def image_callback(self, msg: Image):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        if mask.shape[:2] != (self.mask_h, self.mask_w):
            mask = cv2.resize(
                mask,
                (self.mask_w, self.mask_h),
                interpolation=cv2.INTER_NEAREST
            )  
            
        # --- DYNAMIC HOMOGRAPHY CALCULATION ---
        src_points = self._get_src_points()
        dst_points = self._get_dst_points()
        h_matrix, _ = cv2.findHomography(src_points, dst_points)
        
        if self.get_parameter("calibrate_mode").value:
            calib_img = self._draw_calib(mask, src_points)
            calib_msg = self.bridge.cv2_to_imgmsg(calib_img, "bgr8")
            calib_msg.header = msg.header
            self.pub_bev_grid.publish(calib_msg)

        mask_undist = undistort_mask(mask, self.k, self.d)
        bev_mask = to_bev(
            mask_undist,
            h_matrix,
            bev_size=(self.bev_size, self.bev_size)
        )
        bev_clean = cleanup_bev(bev_mask)  
                   
        bev_msg        = self.bridge.cv2_to_imgmsg(bev_clean, encoding="mono8")
        bev_msg.header = msg.header
        self.pub_bev_mask.publish(bev_msg)  
        
        bev_color   = self.colorize(bev_clean)
        bev_vis_msg = self.bridge.cv2_to_imgmsg(bev_color, encoding="bgr8")
        bev_vis_msg.header = msg.header
        self.pub_bev.publish(bev_vis_msg)

        # Use shared renderer for overlay
        overlay = self.renderer.driving_overlay(
            bev_clean,
            self.analyzer,
            self.lane_width_px,
            self.camera_offset_x_px,
            (self.roi_start_ratio, self.roi_end_ratio),
        )
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)
        
        # --- DEBUG CALCULATION ---
        if self.debug_mode:
            # Make a copy of the overlay that already has the fitted lines
            debug_img = overlay.copy()
            h, w = debug_img.shape[:2]

            # 1. Draw the Static ROI Bounds (Red & Blue)
            y0 = int(round(h * self.roi_start_ratio))
            y1 = int(round(h * self.roi_end_ratio))
            y0 = max(0, min(h - 1, y0))
            y1 = max(y0 + 1, min(h, y1))

            cv2.line(debug_img, (0, y0), (w, y0), (0, 0, 255), 2)  # ROI Start (Red)
            cv2.line(debug_img, (0, y1), (w, y1), (255, 0, 0), 2)  # ROI End (Blue)

            _, _, _, center_y_min, _, _, _ = self.analyzer.analyze(bev_mask)

            # 3. Calculate and Draw the Dynamic Lookahead Target (Green)
            visible_ratio = float(center_y_min) / float(h)
            dynamic_target_ratio = np.clip(visible_ratio + 0.10, 0.95, 0.99)
            target_y_px = int(round(dynamic_target_ratio * float(h - 1)))
            
            # Draw a bright green line for the dynamic target
            cv2.line(debug_img, (0, target_y_px), (w, target_y_px), (0, 0, 0), 2)

            # Convert to ROS Image message and publish
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self.pub_bev_debug.publish(debug_msg)
        
        self._frames += 1
        if self._frames == 1:
            self.get_logger().info(
                f"First mask received - classes in frame: {np.unique(mask).tolist()}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = SegBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()  

if __name__ == "__main__":
    main()
