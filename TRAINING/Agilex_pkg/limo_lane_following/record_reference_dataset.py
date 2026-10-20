#!/usr/bin/env python3
"""Record exactly matched segmentation masks and camera-reference results.

Inputs:
    /seg/mask_raw                  sensor_msgs/Image, mono8
    /lane/metric_reference_debug   std_msgs/String containing JSON
    /lane/camera_reference_debug   sensor_msgs/Image, optional

Matching uses the image header timestamp and the timestamp INSIDE
the reference JSON. There is no approximate-time pairing.

Outputs per matched frame:
    frame_NNNNNN_SEC_NSEC_mask.png
    frame_NNNNNN_SEC_NSEC_data.json
    frame_NNNNNN_SEC_NSEC_debug.png  (if received)

Never publishes motion commands.
"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def image_stamp(msg):
    return int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)


def reference_stamp(data):
    sec = int(data["stamp_sec"])
    nanosec = int(data["stamp_nanosec"])
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("Invalid reference timestamp")
    return sec, nanosec


class ReferenceDatasetRecorder(Node):
    def __init__(self):
        super().__init__("reference_dataset_recorder")
        self.bridge = CvBridge()

        self.declare_parameter("output_dir", "~/limo_reference_dataset")
        self.declare_parameter("session_name", "outer_corridor_right")
        self.declare_parameter("cache_size", 150)

        root = Path(
            os.path.expanduser(
                str(self.get_parameter("output_dir").value)
            )
        )
        session = str(self.get_parameter("session_name").value)

        # Avoid accidentally writing into a previous lap.
        if not session or Path(session).name != session:
            raise ValueError("session_name must be a single directory name")

        self.target = root / session
        if self.target.exists() and any(self.target.iterdir()):
            raise RuntimeError(
                f"{self.target} is not empty. Choose a new session_name."
            )
        self.target.mkdir(parents=True, exist_ok=True)

        self.cache_size = max(
            10, int(self.get_parameter("cache_size").value)
        )
        self.masks = OrderedDict()
        self.references = OrderedDict()
        self.debug_images = OrderedDict()
        self.saved = OrderedDict()
        self.frame_index = 0
        self.evicted_masks = 0
        self.evicted_references = 0
        self.debug_count = 0

        self.create_subscription(
            Image, "/seg/mask_raw", self.on_mask, 20
        )
        self.create_subscription(
            String, "/lane/metric_reference_debug",
            self.on_reference, 20
        )
        self.create_subscription(
            Image, "/lane/camera_reference_debug",
            self.on_debug, 20
        )

        self.get_logger().info(
            f"Recording exact-timestamp pairs to {self.target}"
        )

    def bounded_insert(self, cache, stamp, item, kind):
        cache[stamp] = item
        cache.move_to_end(stamp)

        while len(cache) > self.cache_size:
            cache.popitem(last=False)
            if kind == "mask":
                self.evicted_masks += 1
            elif kind == "reference":
                self.evicted_references += 1

    def on_mask(self, msg):
        stamp = image_stamp(msg)
        if stamp == (0, 0):
            self.get_logger().warn("Mask has zero timestamp; skipping")
            return

        self.bounded_insert(self.masks, stamp, msg, "mask")
        self.try_save(stamp)

    def on_reference(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("Reference JSON is not an object")
            stamp = reference_stamp(data)
            if stamp == (0, 0):
                raise ValueError("Reference has zero source timestamp")
        except (ValueError, TypeError, KeyError) as exc:
            self.get_logger().warn(
                f"Skipping malformed reference: {exc}"
            )
            return

        self.bounded_insert(
            self.references, stamp, data, "reference"
        )
        self.try_save(stamp)

    def on_debug(self, msg):
        stamp = image_stamp(msg)

        # The debug image can arrive before OR after the mask/JSON
        # pair. Save it later if the pair has already been recorded.
        if stamp in self.saved:
            self.save_debug(stamp, self.saved[stamp], msg)
        else:
            self.bounded_insert(
                self.debug_images, stamp, msg, "debug"
            )

    def save_debug(self, stamp, stem, msg):
        path = self.target / f"{stem}_debug.png"
        if path.exists():
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            if not cv2.imwrite(str(path), image):
                raise OSError("cv2.imwrite returned false")
            self.debug_count += 1
        except Exception as exc:
            self.get_logger().warn(
                f"Could not save debug image for {stamp}: {exc}"
            )

    def try_save(self, stamp):
        if stamp not in self.masks or stamp not in self.references:
            return

        msg = self.masks.pop(stamp)
        reference = self.references.pop(stamp)

        try:
            mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8"
            )
            if mask.shape != (240, 320):
                raise ValueError(
                    f"Expected 240x320 mask; got {mask.shape}"
                )

            stem = (
                f"frame_{self.frame_index:06d}_"
                f"{stamp[0]}_{stamp[1]}"
            )
            mask_path = self.target / f"{stem}_mask.png"
            data_path = self.target / f"{stem}_data.json"

            if not cv2.imwrite(str(mask_path), mask):
                raise OSError("Failed to write mask PNG")

            record = {
                "frame_index": self.frame_index,
                "mask_file": mask_path.name,
                "debug_file_if_available": (
                    f"{stem}_debug.png"
                ),
                "mask_stamp_sec": stamp[0],
                "mask_stamp_nanosec": stamp[1],
                "mask_encoding": msg.encoding,
                "mask_shape": list(mask.shape),
                "reference_payload": reference,
            }

            with open(data_path, "w", encoding="utf-8") as fp:
                json.dump(record, fp, indent=2)

        except Exception as exc:
            self.get_logger().error(
                f"Failed to save matched frame {stamp}: {exc}"
            )
            return

        self.saved[stamp] = stem
        while len(self.saved) > self.cache_size:
            self.saved.popitem(last=False)

        debug_msg = self.debug_images.pop(stamp, None)
        if debug_msg is not None:
            self.save_debug(stamp, stem, debug_msg)

        self.frame_index += 1
        if self.frame_index % 25 == 0:
            self.get_logger().info(
                f"Saved {self.frame_index} matched frames; "
                f"{self.debug_count} debug images; "
                f"evicted unmatched masks={self.evicted_masks}, "
                f"references={self.evicted_references}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceDatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Finished: {node.frame_index} matched frames, "
            f"{node.debug_count} debug images. "
            f"Unmatched evictions: masks={node.evicted_masks}, "
            f"references={node.evicted_references}."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()