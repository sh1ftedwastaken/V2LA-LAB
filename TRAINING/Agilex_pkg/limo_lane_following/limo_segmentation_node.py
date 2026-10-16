#!/usr/bin/env python3
"""
limo_segmentation_node.py
=========================
ROS 2 node for running the trained segmentation model on the real robot.

Behaviour
---------
- Subscribes to the camera topic.
- Upscales the incoming image to the configured output size.
- Publishes the upscaled raw frame.
- Runs the trained model and publishes:
  - a BEV-ready single-channel mask on /seg/mask
  - a color label image on /limo/camera/label_image
  - a camera overlay on /seg/cam_overlay (now with polynomial lane curves)
- Optionally saves raw frames and predicted outputs.
"""
from __future__ import annotations #keep this since the limo car has a python version older
import os
import threading
from pathlib import Path
from datetime import datetime
import math

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError

from train2 import build_model

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
)


NUM_CLASSES = 5
CLASS_COLORS = np.array([
    [0, 255, 0],
    [100, 100, 100],
    [255, 255, 255],
    [255, 255, 0],
    [0, 0, 255],
], dtype=np.uint8)


def load_model(weights_path: str, device: torch.device, model_name: str, encoder_weights: str):
    checkpoint = torch.load(weights_path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    resolved_model = model_name or checkpoint_args.get("model", "lightunet")
    resolved_encoder_weights = encoder_weights or checkpoint_args.get("encoder_weights", "imagenet")
    resolved_encoder_weights = None if resolved_encoder_weights == "none" else resolved_encoder_weights
    dropout_p = checkpoint_args.get("dropout_p", 0.3)

    model = build_model(
        resolved_model,
        dropout_p=dropout_p,
        encoder_weights=resolved_encoder_weights,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, resolved_model


class LimoSegmentationNode(Node):

    def __init__(self):
        super().__init__("limo_segmentation_node")

        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/limo/camera/image_upscaled")
        self.declare_parameter("mask_topic", "/seg/mask")
        self.declare_parameter("mask_raw_topic", "/seg/mask_raw")
        self.declare_parameter("label_topic", "/limo/camera/label_image")
        self.declare_parameter("overlay_topic", "/seg/cam_overlay")
        self.declare_parameter("calibration_topic", "/seg/calibration_grid")
        self.declare_parameter("save_dir", os.path.expanduser("~/limo_frames"))
        self.declare_parameter("save_every", 1)
        self.declare_parameter("save_frames", False)
        self.declare_parameter("segment_every", 1)
        self.declare_parameter("target_width", 640)
        self.declare_parameter("target_height", 480)
        self.declare_parameter("model_input_width", 320)
        self.declare_parameter("model_input_height", 240)
        self.declare_parameter("bev_mask_width", 320)
        self.declare_parameter("bev_mask_height", 240)
        self.declare_parameter("pad_to_multiple", 32)
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("model_name", "")
        self.declare_parameter("encoder_weights", "")
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("save_predictions", False)
        self.declare_parameter("overlay_alpha", 0.45)
        self.declare_parameter("calibrate_mode", False)


        # Camera overlay parameters (new)
        self.declare_parameter("camera_overlay_enabled", False)
        self.declare_parameter("camera_roi_start_ratio", 0.75)
        self.declare_parameter("camera_roi_end_ratio", 0.95)
        self.declare_parameter("camera_lane_width_px", 100.0)
        self.declare_parameter("camera_offset_x_px", 0.0)
        

        # BEV Source Parameters
        self.declare_parameter("bev_src_bottom_left_x", 11.0)
        self.declare_parameter("bev_src_bottom_left_y", 238.0)      
        self.declare_parameter("bev_src_bottom_right_x", 305.5)     
        self.declare_parameter("bev_src_bottom_right_y", 238.0)
        self.declare_parameter("bev_src_top_right_x", 210.0)
        self.declare_parameter("bev_src_top_right_y", 160.0)
        self.declare_parameter("bev_src_top_left_x", 105.0)
        self.declare_parameter("bev_src_top_left_y", 160.0)
        
        # BEV Destination Parameters (Margin-based)
        self.declare_parameter("bev_dst_margin_x", 20.0)
        self.declare_parameter("bev_dst_top_y", 5.0)
        self.declare_parameter("bev_dst_bottom_y", 162)

        cam_topic         = self.get_parameter("camera_topic").value
        out_topic         = self.get_parameter("output_topic").value
        mask_topic        = self.get_parameter("mask_topic").value
        mask_raw_topic    = self.get_parameter("mask_raw_topic").value
        label_topic       = self.get_parameter("label_topic").value
        overlay_topic     = self.get_parameter("overlay_topic").value
        calibration_topic = self.get_parameter("calibration_topic").value

        self.save_dir         = os.path.expanduser(self.get_parameter("save_dir").value)
        self.save_every       = self.get_parameter("save_every").value
        self.save_frames      = self.get_parameter("save_frames").value
        self.segment_every    = self.get_parameter("segment_every").value
        self.tgt_w            = self.get_parameter("target_width").value
        self.tgt_h            = self.get_parameter("target_height").value
        self.model_w          = self.get_parameter("model_input_width").value
        self.model_h          = self.get_parameter("model_input_height").value
        self.bev_mask_w       = self.get_parameter("bev_mask_width").value
        self.bev_mask_h       = self.get_parameter("bev_mask_height").value
        self.pad_to_multiple  = self.get_parameter("pad_to_multiple").value
        self.publish_overlay  = self.get_parameter("publish_overlay").value
        self.save_predictions = self.get_parameter("save_predictions").value
        self.calibrate_mode = self.get_parameter("calibrate_mode").value
        self.overlay_alpha    = float(self.get_parameter("overlay_alpha").value)
        checkpoint            = self.get_parameter("checkpoint").value
        model_name            = self.get_parameter("model_name").value
        encoder_weights       = self.get_parameter("encoder_weights").value

        # Camera overlay params
        self.camera_overlay_enabled = self.get_parameter("camera_overlay_enabled").value
        self.camera_roi_start_ratio = self.get_parameter("camera_roi_start_ratio").value
        self.camera_roi_end_ratio   = self.get_parameter("camera_roi_end_ratio").value
        self.camera_lane_width_px   = self.get_parameter("camera_lane_width_px").value
        self.camera_offset_x_px     = self.get_parameter("camera_offset_x_px").value

        if not checkpoint:
            raise ValueError("checkpoint parameter must point to a trained model.")

        self.bridge = CvBridge()
        self.frame_count = 0
        self.saved_count = 0
        self._lock = threading.Lock()
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.model, self.resolved_model = load_model(
            weights_path=checkpoint,
            device=self.device,
            model_name=model_name,
            encoder_weights=encoder_weights,
        )

        self.pub_image            = self.create_publisher(RosImage, out_topic, 2)
        self.pub_mask             = self.create_publisher(RosImage, mask_topic, 2)
        self.pub_mask_raw         = self.create_publisher(RosImage, mask_raw_topic, 2)
        self.pub_label            = self.create_publisher(RosImage, label_topic, 2)
        self.pub_saved            = self.create_publisher(Bool, "/limo/camera/frame_saved", 2)
        self.pub_status           = self.create_publisher(String, "/limo/camera/status", 2)
        self.pub_overlay          = None
        if self.publish_overlay:
            self.pub_overlay = self.create_publisher(RosImage, overlay_topic, 2)
        self.pub_calibration_grid = self.create_publisher(RosImage, calibration_topic, 2)

        # Initialize shared lane analyzer and renderer for camera overlay
        self.cam_analyzer = LaneAnalyzer(
            lane_width_px=self.camera_lane_width_px,
            camera_offset_x_px=self.camera_offset_x_px,
            roi_start_ratio=self.camera_roi_start_ratio,
            roi_end_ratio=self.camera_roi_end_ratio,
            alpha_lane_width=0.05,
        )
        self.cam_renderer = LaneOverlayRenderer()

        self.sub = self.create_subscription(RosImage, cam_topic, self._image_callback, 1)
        self.create_timer(1.0, self._heartbeat)

        self.get_logger().info(f"Subscribed to   : {cam_topic}")
        self.get_logger().info(f"Checkpoint      : {checkpoint}")
        self.get_logger().info(f"Detected model  : {self.resolved_model}")
        self.get_logger().info(f"Device          : {self.device}")
        self.get_logger().info(f"Mask topic      : {mask_topic} (viewer-friendly)")
        self.get_logger().info(f"Raw mask topic  : {mask_raw_topic} (mono8 for BEV)")
        self.get_logger().info(f"Output size     : {self.tgt_w}x{self.tgt_h}")
        self.get_logger().info(f"Model input     : {self.model_w}x{self.model_h}")
        self.get_logger().info(f"Pad multiple    : {self.pad_to_multiple}")
        self.get_logger().info(f"BEV mask size   : {self.bev_mask_w}x{self.bev_mask_h}")
        self.get_logger().info(f"Camera overlay  : {'enabled' if self.camera_overlay_enabled else 'disabled'}")
        self.get_logger().info(f"Calibration grid: {'enabled' if self.calibrate_mode else 'disabled'}")
        self.get_logger().info("Node ready - waiting for images...")

    def _pad_image_for_model(self, image_rgb: np.ndarray):
        multiple      = max(1, int(self.pad_to_multiple))
        height, width = image_rgb.shape[:2]
        padded_h      = int(math.ceil(height / multiple) * multiple)
        padded_w      = int(math.ceil(width / multiple) * multiple)

        pad_top    = (padded_h - height) // 2
        pad_bottom = padded_h - height - pad_top
        pad_left   = (padded_w - width) // 2
        pad_right  = padded_w - width - pad_left

        padded = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return padded, (pad_top, pad_bottom, pad_left, pad_right)

    def _predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        padded, (pad_top, pad_bottom, pad_left, pad_right) = self._pad_image_for_model(resized)
        pil_image = Image.fromarray(padded)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            predicted = self.model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if pad_top or pad_bottom or pad_left or pad_right:
            h_end = predicted.shape[0] - pad_bottom if pad_bottom > 0 else predicted.shape[0]
            w_end = predicted.shape[1] - pad_right if pad_right > 0 else predicted.shape[1]
            predicted = predicted[pad_top:h_end, pad_left:w_end]

        mask = cv2.resize(predicted, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        color_rgb = CLASS_COLORS[np.clip(mask, 0, NUM_CLASSES - 1)]
        return cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    
    def _draw_calibration(self, image: np.ndarray, src_points: np.ndarray) -> np.ndarray:
        """Draws the calibration trapezoid over the camera image."""
        vis = image.copy()
        pts = src_points.astype(np.int32).reshape(-1, 1, 2)
        
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 255), thickness=1)
        
        labels = ["0:BL", "1:BR", "2:TR", "3:TL"]
        for (x, y), lbl in zip(src_points.astype(int), labels):
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(vis, lbl, (x + 4, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        return vis  

    def _blend_overlay(self, image_bgr: np.ndarray, color_bgr: np.ndarray) -> np.ndarray:
        alpha = max(0.0, min(1.0, self.overlay_alpha))
        overlay = (1.0 - alpha) * image_bgr.astype(np.float32) + alpha * color_bgr.astype(np.float32)
        return overlay.clip(0, 255).astype(np.uint8)

    def _resize_mask_for_bev(self, mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask, (self.bev_mask_w, self.bev_mask_h), interpolation=cv2.INTER_NEAREST)
    
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

    def _image_callback(self, msg: RosImage):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        self.frame_count += 1
        upscaled = cv2.resize(img_bgr, (self.tgt_w, self.tgt_h), interpolation=cv2.INTER_CUBIC)

        try:
            raw_msg = self.bridge.cv2_to_imgmsg(upscaled, encoding="bgr8")
            raw_msg.header = msg.header
            self.pub_image.publish(raw_msg)
        except CvBridgeError as exc:
            self.get_logger().error(f"Raw publish error: {exc}")

        if self.frame_count % self.segment_every == 0:
            try:
                full_mask = self._predict_mask(upscaled)
                label_bgr = self._colorize_mask(full_mask)
                bev_mask = self._resize_mask_for_bev(full_mask)
                overlay_bgr = self._blend_overlay(upscaled, label_bgr)

                mask_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
                mask_msg.header = msg.header
                self.pub_mask.publish(mask_msg)

                mask_raw_msg = self.bridge.cv2_to_imgmsg(bev_mask, encoding="mono8")
                mask_raw_msg.header = msg.header
                self.pub_mask_raw.publish(mask_raw_msg)

                label_msg = self.bridge.cv2_to_imgmsg(label_bgr, encoding="bgr8")
                label_msg.header = msg.header
                self.pub_label.publish(label_msg)

                # Publish camera overlay with polynomial lane curves
                if self.pub_overlay is not None and self.camera_overlay_enabled and full_mask is not None:
                    cam_overlay = self.cam_renderer.driving_overlay(
                        full_mask,
                        self.cam_analyzer,
                        self.camera_lane_width_px,
                        self.camera_offset_x_px,
                        (self.camera_roi_start_ratio, self.camera_roi_end_ratio),
                    )
                    overlay_msg = self.bridge.cv2_to_imgmsg(cam_overlay, encoding="bgr8")
                    overlay_msg.header = msg.header
                    self.pub_overlay.publish(overlay_msg)
                    
                # --- CALIBRATION MODE ---
                if self.calibrate_mode:
                    calibration_img = cv2.resize(
                        img_bgr,
                        (self.model_w, self.model_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    src_points = self._get_src_points()
                    calibration_img = self._draw_calibration(calibration_img, src_points)
                    calibration_msg = self.bridge.cv2_to_imgmsg(calibration_img, encoding="bgr8")
                    calibration_msg.header = msg.header
                    self.pub_calibration_grid.publish(calibration_msg) 
                    
            except Exception as exc:
                self.get_logger().warn(f"Segmentation failed: {exc}")
                full_mask   = None
                label_bgr   = None
                overlay_bgr = None
        else:
            full_mask   = None
            label_bgr   = None
            overlay_bgr = None

        if self.save_frames and self.frame_count % self.save_every == 0:
            self._save_outputs(upscaled, full_mask, label_bgr, overlay_bgr)

    def _save_outputs(self, raw_bgr: np.ndarray, mask: np.ndarray | None, label_bgr: np.ndarray | None, overlay_bgr: np.ndarray | None):
        with self._lock:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"frame_{self.saved_count:06d}_{ts}"
            raw_path = os.path.join(self.save_dir, f"{stem}.png")
            cv2.imwrite(raw_path, raw_bgr)

            if self.save_predictions and mask is not None:
                cv2.imwrite(os.path.join(self.save_dir, f"{stem}_mask.png"), mask)
                if label_bgr is not None:
                    cv2.imwrite(os.path.join(self.save_dir, f"{stem}_label.png"), label_bgr)
                if overlay_bgr is not None:
                    cv2.imwrite(os.path.join(self.save_dir, f"{stem}_overlay.png"), overlay_bgr)

            self.saved_count += 1

        self.pub_saved.publish(Bool(data=True))
        self.pub_status.publish(String(data=f"saved:{raw_path}|total:{self.saved_count}"))
        self.get_logger().info(f"Saved [{self.saved_count:>6}]: {Path(raw_path).name}")

    def _heartbeat(self):
        self.pub_status.publish(
            String(data=f"running|frames:{self.frame_count}|saved:{self.saved_count}")
        )


def main(args=None):
    rclpy.init(args=args)
    node = LimoSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()