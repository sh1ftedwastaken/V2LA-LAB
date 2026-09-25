#!/usr/bin/env python3
"""
lane_analyzer.py
================

Shared lane analysis and overlay-rendering module.

This module has no ROS dependencies. It is used by:
- seg_bev_node.py
- autonomous_driving.py
- limo_segmentation_node.py

Important behavior
------------------
- Yellow and white are segmentation classes only.
- Color does NOT determine whether a boundary is left or right.
- Boundary side is determined from the fitted path's closest observed point.
- Lane paths are represented as parametric [x, y] waypoints.
- Single boundaries are shifted along their local perpendicular normals.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# SEGMENTATION CLASSES
# =========================================================

CLASS_BG = 0
CLASS_ROAD = 1
CLASS_WHITE = 2
CLASS_YELLOW = 3
CLASS_VEHICLE = 4

# =========================================================
# LANE ANALYSIS DEFAULTS
# =========================================================

DEFAULT_LANE_WIDTH_PX = 128.0

MIN_POLYFIT_POINTS = 30
MIN_OBSTACLE_AREA_PX = 8

DEFAULT_ROI_START_RATIO = 0.70
DEFAULT_ROI_END_RATIO = 0.95

MIN_BOTH_LANE_GAP_PX = 35.0
MAX_COMPONENT_ROI_DISTANCE_PX = 28.0

LANE_BIN_SIZE_PX = 5
MORPHOLOGY_KERNEL_SIZE = 15
PATH_POINT_COUNT = 150

# Keep the current extrapolation behavior. The observed line corresponds
# approximately to t=[0, 1], while the extra range supports the blind spot.
PATH_T_MIN = -1.0
PATH_T_MAX = 2.0

# =========================================================
# POLYFIT ERROR HANDLING
# =========================================================

_RANK_WARNING_TYPES = tuple(
    warning_type
    for warning_type in (
        getattr(np, "RankWarning", None),
        getattr(getattr(np, "exceptions", None), "RankWarning", None),
    )
    if warning_type is not None
)

_POLYFIT_ERRORS = (
    np.linalg.LinAlgError,
    ValueError,
    TypeError,
    FloatingPointError,
) + _RANK_WARNING_TYPES


# =========================================================
# DRAWING COLORS
# =========================================================

CLASS_COLORS = {
    CLASS_BG:      (  0, 255,   0),
    CLASS_ROAD:    (100, 100, 100),
    CLASS_WHITE:   (255, 255, 255),
    CLASS_YELLOW:  (  0, 255, 255),
    CLASS_VEHICLE: (  0,   0, 255),
}

CURVE_YELLOW = (255,   0, 255)
CURVE_WHITE  = (255, 128,   0)
CURVE_CENTER = (255,   0,   0)
OBSTACLE_BOX = (  0,  30, 255)
EGO_AXIS     = (255, 210,  40)


# =========================================================
# ANALYZER STATES
# =========================================================

STATE_BOTH       = "BOTH_LANES"
STATE_LEFT_ONLY  = "LEFT_LANE_ONLY"
STATE_RIGHT_ONLY = "RIGHT_LANE_ONLY"
STATE_LOST       = "LOST"
STATE_DRIVABLE   = "DRIVABLE_AREA_MODE"


# =========================================================
# DATA STRUCTURES
# =========================================================

class LaneFit(NamedTuple):
    """Parametric lane path and its measurement metadata."""

    path_points: np.ndarray
    y_min: int
    y_max: int
    roi_x: float
    count: int
    rmse: float


class BoundaryCandidate(NamedTuple):
    """A fitted physical boundary classified relative to the robot."""

    color_name: str
    fit: LaneFit
    side: str
    anchor_x: float


# =========================================================
# LANE ANALYZER
# =========================================================

class LaneAnalyzer:
    """
    Shared parametric lane analyzer.

    Yellow and white classes are fitted independently, but their colors are
    not used to determine whether they are left or right boundaries.

    Left/right classification is based on the fitted path's X coordinate at
    the closest observed Y coordinate.
    """

    def __init__(
        self,
        lane_width_px: float = DEFAULT_LANE_WIDTH_PX,
        camera_offset_x_px: float = 0.0,
        roi_start_ratio: float = DEFAULT_ROI_START_RATIO,
        roi_end_ratio: float = DEFAULT_ROI_END_RATIO,
        alpha_lane_width: float = 0.05,
    ):
        self.lane_width_px = float(lane_width_px)
        self.camera_offset_x_px = float(camera_offset_x_px)
        self.roi_start_ratio = float(np.clip(roi_start_ratio, 0.0, 1.0))
        self.roi_end_ratio = float(np.clip(roi_end_ratio, 0.0, 1.0))

        if self.roi_end_ratio <= self.roi_start_ratio:
            self.roi_end_ratio = min(1.0,self.roi_start_ratio + 0.05)

        self.alpha_lane_width = float(np.clip(alpha_lane_width, 0.0, 1.0))

    # -----------------------------------------------------
    # ROI and measurements
    # -----------------------------------------------------

    def roi_bounds(self, mask: np.ndarray) -> Tuple[int, int]:
        """Return the vertical [y0, y1) bounds of the detection ROI."""

        h, _ = mask.shape

        y0 = int(round(h * self.roi_start_ratio))
        y1 = int(round(h * self.roi_end_ratio))

        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        return y0, y1

    def lane_measurement(
        self,
        roi: np.ndarray,
        cls: int,
    ) -> Tuple[Optional[float], int]:
        """
        Measure the median X position of one segmentation class in the ROI.

        Returns:
            (median_x, pixel_count), or (None, 0) if insufficient evidence.
        """

        _, xs = np.where(roi == cls)

        if xs.size < 12:
            return None, 0

        return float(np.median(xs)), int(xs.size)

    # -----------------------------------------------------
    # Connected-component extraction
    # -----------------------------------------------------

    def _selected_component_points(
        self,
        mask: np.ndarray,
        cls: int,
        roi_x: float,
        y0: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the connected component nearest the measured ROI position.

        Morphological closing bridges small segmentation gaps before connected
        components are computed.
        """

        class_mask = (mask == cls).astype(np.uint8)

        kernel_size = max(3, int(MORPHOLOGY_KERNEL_SIZE))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        class_mask = cv2.morphologyEx(
            class_mask, cv2.MORPH_CLOSE, kernel,
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
        )

        best_label = None
        best_distance = float("inf")
        best_roi_count = 0

        roi_labels = labels[y0:y1, :]

        for label in range(1, num_labels):
            component_area = int(stats[label, cv2.CC_STAT_AREA])

            if component_area <= MIN_POLYFIT_POINTS:
                continue

            _, roi_xs = np.where(roi_labels == label)

            if roi_xs.size < 12:
                continue

            component_roi_x = float(np.median(roi_xs))
            distance = abs(component_roi_x - float(roi_x))

            if distance < best_distance:
                best_distance = distance
                best_roi_count = int(roi_xs.size)
                best_label = label

        if best_label is None:
            return (
                np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            )

        if best_distance > MAX_COMPONENT_ROI_DISTANCE_PX:
            return (
                np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            )

        ys, xs = np.where(labels == best_label)

        minimum_count = max(MIN_POLYFIT_POINTS, best_roi_count)

        if ys.size < minimum_count:
            return (
                np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            )

        return ys, xs

    def _fit_selected_lane(
        self,
        mask: np.ndarray,
        cls: int,
        roi_x: Optional[float],
        count: int,
        y0: int,
        y1: int,
    ) -> Optional[LaneFit]:
        """Extract and fit the selected component for one class."""

        if roi_x is None:
            return None

        lane_y, lane_x = self._selected_component_points(
            mask, cls, roi_x, y0, y1
        )

        if lane_y.size == 0:
            return None

        return self._fit_lane_poly(lane_y, lane_x, roi_x, count)

    # -----------------------------------------------------
    # Parametric path fitting
    # -----------------------------------------------------

    def _fit_lane_poly(
        self,
        y_points: np.ndarray,
        x_points: np.ndarray,
        roi_x: float,
        count: int,
    ) -> Optional[LaneFit]:
        """
        Fit a parametric lane path x(t), y(t).

        Dynamic-axis binning is used:
        - Mostly vertical components are binned by Y.
        - Mostly horizontal components are binned by X.

        The resulting center points are ordered from the end closest to the
        robot toward the farther end.
        """

        if y_points.size <= MIN_POLYFIT_POINTS:
            return None

        y_points = y_points.astype(np.float64)
        x_points = x_points.astype(np.float64)

        y_span = float(np.max(y_points) - np.min(y_points))
        x_span = float(np.max(x_points) - np.min(x_points))

        bin_size = max(1, int(LANE_BIN_SIZE_PX))

        if y_span >= x_span:
            # Mostly vertical path: bin by Y and compute median X.
            bins = (y_points // bin_size).astype(np.int64)
            unique_bins = np.unique(bins)

            sy = np.zeros(len(unique_bins), dtype=np.float64)
            sx = np.zeros(len(unique_bins), dtype=np.float64)

            for i, bin_id in enumerate(unique_bins):
                selected = bins == bin_id
                sy[i] = np.median(y_points[selected])
                sx[i] = np.median(x_points[selected])

            # Image Y increases downward. Start at the closest end.
            order = np.argsort(sy)[::-1]

        else:
            # Mostly horizontal path: bin by X and compute median Y.
            bins = (x_points // bin_size).astype(np.int64)
            unique_bins = np.unique(bins)

            sx = np.zeros(len(unique_bins), dtype=np.float64)
            sy = np.zeros(len(unique_bins), dtype=np.float64)

            for i, bin_id in enumerate(unique_bins):
                selected = bins == bin_id
                sx[i] = np.median(x_points[selected])
                sy[i] = np.median(y_points[selected])

            left_index = int(np.argmin(sx))
            right_index = int(np.argmax(sx))

            # Begin from whichever horizontal end is closer to the car.
            if sy[left_index] > sy[right_index]:
                order = np.argsort(sx)
            else:
                order = np.argsort(sx)[::-1]

        sx = sx[order]
        sy = sy[order]

        if len(sx) < 3:
            return None

        segment_lengths = np.hypot(
            np.diff(sx),
            np.diff(sy),
        )

        # Remove exact duplicate consecutive points.
        keep = np.ones(len(sx), dtype=bool)
        keep[1:] = segment_lengths > 1e-6

        sx = sx[keep]
        sy = sy[keep]

        if len(sx) < 3:
            return None

        segment_lengths = np.hypot(
            np.diff(sx),
            np.diff(sy),
        )

        t = np.zeros(len(sx), dtype=np.float64)
        t[1:] = np.cumsum(segment_lengths)

        if t[-1] <= 1e-6:
            return None

        t_norm = t / t[-1]

        degree_x = 2 if len(t_norm) > 3 else 1

        # Keep Y monotonic to prevent backward hooks that can flip normals.
        degree_y = 1

        try:
            with warnings.catch_warnings():
                for warning_type in _RANK_WARNING_TYPES:
                    warnings.simplefilter("error", warning_type)

                coeffs_x = np.polyfit(t_norm, sx, degree_x)
                coeffs_y = np.polyfit(t_norm, sy, degree_y)

        except _POLYFIT_ERRORS:
            return None

        if not (
            np.all(np.isfinite(coeffs_x)) and np.all(np.isfinite(coeffs_y))
        ):
            return None

        t_eval = np.linspace(PATH_T_MIN, PATH_T_MAX, PATH_POINT_COUNT)

        path_x = np.polyval(coeffs_x, t_eval)
        path_y = np.polyval(coeffs_y, t_eval)

        path_points = np.column_stack((path_x, path_y)).astype(np.float64)

        path_points[:, 0] += float(self.camera_offset_x_px)

        if not np.isfinite(path_points).all():
            return None

        # Compute a simple parametric fit error over the observed points.
        predicted_x = np.polyval(coeffs_x, t_norm)
        predicted_y = np.polyval(coeffs_y, t_norm)

        rmse = float(
            np.sqrt(np.mean((predicted_x - sx) ** 2 + (predicted_y - sy) ** 2))
        )

        return LaneFit(
            path_points=path_points,
            y_min=int(np.floor(np.min(sy))),
            y_max=int(np.ceil(np.max(sy))),
            roi_x=float(roi_x),
            count=int(count),
            rmse=rmse,
        )

    # -----------------------------------------------------
    # Bottom-anchor boundary classification
    # -----------------------------------------------------

    def _bottom_anchor_x(self, fit: LaneFit) -> float:
        """
        Return the path X coordinate at the closest observed line position.

        The fitted path is extrapolated beyond the observed component. Using
        argmax(path_y) directly could select an artificial extrapolated point.

        Therefore, this method finds the path point closest to fit.y_max,
        which is the bottom of the actual observed component.
        """

        points = fit.path_points

        if points is None or len(points) == 0:
            return float(fit.roi_x)

        finite = np.isfinite(points).all(axis=1)

        if not np.any(finite):
            return float(fit.roi_x)

        valid_points = points[finite]

        index = int(
            np.argmin(np.abs(valid_points[:, 1] - float(fit.y_max)))
        )

        return float(valid_points[index, 0])

    def _make_boundary_candidate(
        self,
        color_name: str,
        fit: LaneFit,
        ego_center: float,
    ) -> BoundaryCandidate:
        """
        Convert a color fit into a physical left/right boundary candidate.
        """

        anchor_x = self._bottom_anchor_x(fit)

        side = ("left" if anchor_x < ego_center else "right")

        return BoundaryCandidate(
            color_name=color_name,
            fit=fit,
            side=side,
            anchor_x=anchor_x,
        )

    @staticmethod
    def _candidate_strength(
        candidate: BoundaryCandidate,
    ) -> Tuple[int, float]:
        """
        Rank candidates using component evidence.

        More pixels are preferred. If counts match, lower RMSE is preferred.
        """

        return (int(candidate.fit.count), -float(candidate.fit.rmse))

    # -----------------------------------------------------
    # Path geometry
    # -----------------------------------------------------

    def _average_lane_paths(
        self,
        left_fit: LaneFit,
        right_fit: LaneFit,
    ) -> np.ndarray:
        """
        Average aligned parametric waypoints from opposite boundaries.
        """

        left_points = left_fit.path_points
        right_points = right_fit.path_points

        point_count = min(len(left_points), len(right_points)) 
        if point_count == 0:
            return np.empty((0, 2), dtype=np.float64)

        return (left_points[:point_count] + right_points[:point_count]) / 2.0

    def _shift_lane_path(
        self,
        fit: LaneFit,
        offset_px: float,
    ) -> np.ndarray:
        """
        Shift a boundary along its local perpendicular normal.

        Positive offset shifts toward image-right for a bottom-to-top path.
        Negative offset shifts toward image-left.
        """

        points = fit.path_points.copy()

        if len(points) < 2:
            return points

        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])

        lengths = np.hypot(dx, dy)
        lengths = np.maximum(lengths, 1e-6)

        # In image coordinates, Y increases downward. For a path ordered
        # from near to far, (-dy, dx) is its right-facing normal.
        normal_x = -dy / lengths
        normal_y = dx / lengths

        points[:, 0] += float(offset_px) * normal_x
        points[:, 1] += float(offset_px) * normal_y

        return points
    
    def _is_path_straight(self, path_points: np.ndarray, threshold: float = 4.5) -> bool:
        """Check if a path is mathematically straight by measuring its bow distance."""
        if path_points is None or len(path_points) < 3:
            return True
            
        start_pt = path_points[0]
        end_pt = path_points[-1]
        mid_idx = len(path_points) // 2
        actual_mid_pt = path_points[mid_idx]
        
        straight_mid_pt = (start_pt + end_pt) / 2.0
        bow_distance = float(np.linalg.norm(actual_mid_pt - straight_mid_pt))
        
        return bow_distance < threshold

    def update_lane_width(
        self,
        measured_width: float,
    ) -> None:
        """Update lane width using an exponential moving average."""

        if not np.isfinite(measured_width):
            return

        if measured_width < MIN_BOTH_LANE_GAP_PX:
            return

        self.lane_width_px = (
            self.alpha_lane_width * float(measured_width)
            + (1.0 - self.alpha_lane_width)
            * float(self.lane_width_px)
        )
        
        # --- NEW: Debug print for manual physical measurement comparison ---
        print(f"[LaneAnalyzer] Detected anchor gap: {measured_width:.2f} px | Updated EMA Lane Width: {self.lane_width_px:.2f} px")

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def analyze(
        self,
        mask: np.ndarray,
    ) -> Tuple[
        Optional[LaneFit],
        Optional[LaneFit],
        Optional[np.ndarray],
        int,
        int,
        str,
        float,
    ]:
        """
        Analyze a segmentation mask.

        Returns:
            yellow_fit,
            white_fit,
            center_path,
            center_y_min,
            center_y_max,
            state,
            multiplier

        Yellow/white are retained in the return value for visualization.
        Centerline generation itself is based on physical left/right boundary
        classification, not color.
        """

        if mask is None or mask.ndim != 2:
            return (None, None, None, 0, 0,STATE_LOST, 1.0)

        h, w = mask.shape
        y0, y1 = self.roi_bounds(mask)
        roi = mask[y0:y1, :]

        ego_center = float(w / 2.0)
        half_lane = float(self.lane_width_px) / 2.0
        # single_line_half_lane = half_lane * 0.87    # REMOVE FOR NOW


        # -------------------------------------------------
        # 1. Measure yellow and white evidence in the ROI.
        # -------------------------------------------------

        yellow_x, yellow_count = self.lane_measurement(
            roi, CLASS_YELLOW,
        )
        white_x, white_count = self.lane_measurement(
            roi, CLASS_WHITE,
        )

        use_yellow = yellow_x is not None
        use_white = white_x is not None

        # If yellow and white measurements are too close, they probably
        # describe the same physical line with mixed segmentation colors.
        if use_yellow and use_white:
            measured_gap = abs(float(white_x) - float(yellow_x))

            if measured_gap < MIN_BOTH_LANE_GAP_PX:
                # Keep only the stronger physical component.
                if yellow_count >= white_count:
                    use_white = False
                else:
                    use_yellow = False

        # -------------------------------------------------
        # 2. Fit selected color components.
        # -------------------------------------------------

        yellow_fit = self._fit_selected_lane(
            mask,
            CLASS_YELLOW,
            yellow_x if use_yellow else None,
            yellow_count,
            y0,
            y1,
        )

        white_fit = self._fit_selected_lane(
            mask,
            CLASS_WHITE,
            white_x if use_white else None,
            white_count,
            y0,
            y1,
        )

        # -------------------------------------------------
        # 3. Convert color fits into physical boundaries.
        # -------------------------------------------------

        candidates = []

        if yellow_fit is not None:
            candidates.append(
                self._make_boundary_candidate(
                    "yellow", yellow_fit, ego_center,
                )
            )

        if white_fit is not None:
            candidates.append(
                self._make_boundary_candidate(
                    "white", white_fit, ego_center
                )
            )

        left_candidates = [
            candidate for candidate in candidates if candidate.side == "left"
        ]

        right_candidates = [
            candidate for candidate in candidates if candidate.side == "right"
        ]
        
        # -------------------------------------------------
        # 4. Evaluate Straightness & Solve Centerline
        # -------------------------------------------------
        center_path = None
        center_y_min = y0
        center_y_max = y1 - 1
        state = STATE_LOST
        tune_offset_px = 0.0

        # Step A: Decide if the road is straight using the strongest visible boundary
        is_straight = True
        if candidates:
            strongest_candidate = max(candidates, key=self._candidate_strength)
            is_straight = self._is_path_straight(strongest_candidate.fit.path_points)

        # Step B: Straight Road Logic (Use both lines if available)
        if is_straight and left_candidates and right_candidates:
            left_candidate = max(left_candidates, key=self._candidate_strength)
            right_candidate = max(right_candidates, key=self._candidate_strength)

            anchor_gap = (right_candidate.anchor_x - left_candidate.anchor_x)

            if anchor_gap >= MIN_BOTH_LANE_GAP_PX:
                left_fit = left_candidate.fit
                right_fit = right_candidate.fit

                center_path = self._average_lane_paths(left_fit, right_fit)

                center_y_min = max(left_fit.y_min, right_fit.y_min)
                center_y_max = min(left_fit.y_max, right_fit.y_max)

                if center_y_min > center_y_max:
                    center_y_min = min(left_fit.y_min, right_fit.y_min)
                    center_y_max = max(left_fit.y_max, right_fit.y_max)

                # Update the dynamic track width ONLY on straightaways
                self.update_lane_width(anchor_gap)

                state = STATE_BOTH
                
        # Step C: Curve Logic or Single-Line Logic
        if center_path is None and candidates:
            # Trust the strongest line
            selected = max(candidates, key=self._candidate_strength)
            selected_fit = selected.fit

            # 1. DRAW PURE GEOMETRIC CENTER (Always exactly 1.0 * half_lane)
            # This ensures the blue line is always visually perfect in the BEV image
            if selected.side == "left":
                center_path = self._shift_lane_path(selected_fit, +half_lane)
                state = STATE_LEFT_ONLY
            else:
                center_path = self._shift_lane_path(selected_fit, -half_lane)
                state = STATE_RIGHT_ONLY

            # 2. CALCULATE CONTROL TUNING OFFSET
            if selected.color_name == "yellow":
                tune_offset_px = (+0.40 * half_lane + 25) if selected.side == "left" else (-0.40 * half_lane - 25)
            else:
                tune_offset_px = (-0.35 * half_lane) if selected.side == "right" else (+0.35 * half_lane)
                
            center_y_min = selected_fit.y_min
            center_y_max = selected_fit.y_max

        # -------------------------------------------------
        # 5. Conservative drivable-area fallback.
        # -------------------------------------------------
        if center_path is None:
            road_ys, road_xs = np.where(roi == CLASS_ROAD)

            if road_xs.size > 50:
                median_x = float(np.median(road_xs))
                center_path = np.column_stack((
                    np.full(50, median_x, dtype=np.float64),
                    np.linspace(center_y_max, center_y_min, 50, dtype=np.float64),
                ))
                state = STATE_DRIVABLE

        # -------------------------------------------------
        # 6. Prevent path from extrapolating beyond visible road
        # -------------------------------------------------
        if center_path is not None and len(center_path) > 0:
            valid_idx = center_path[:, 1] >= (center_y_min - 10) 
            center_path = center_path[valid_idx]
            
            if len(center_path) < 2:
                center_path = None

        return (
            yellow_fit,
            white_fit,
            center_path,
            center_y_min,
            center_y_max,
            state,
            tune_offset_px,
        )


# =========================================================
# LANE OVERLAY RENDERER
# =========================================================

class LaneOverlayRenderer:
    """Stateless renderer for lane-analysis debugging."""

    def __init__(
        self,
        curve_yellow: Tuple[int, int, int] = CURVE_YELLOW,
        curve_white: Tuple[int, int, int] = CURVE_WHITE,
        curve_center: Tuple[int, int, int] = CURVE_CENTER,
        obstacle_box: Tuple[int, int, int] = OBSTACLE_BOX,
        ego_axis: Tuple[int, int, int] = EGO_AXIS,
    ):
        self.curve_yellow = curve_yellow
        self.curve_white = curve_white
        self.curve_center = curve_center
        self.obstacle_box = obstacle_box
        self.ego_axis = ego_axis

    def colorize(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Convert a class mask into a BGR image."""

        color = np.zeros(
            (mask.shape[0], mask.shape[1], 3),
            dtype=np.uint8,
        )

        for cls, bgr in CLASS_COLORS.items():
            color[mask == cls] = bgr

        return color

    def draw_path(
        self,
        overlay: np.ndarray,
        path_points: Optional[np.ndarray],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw a parametric path.

        Points are not clipped directly to the image borders. Direct clipping
        can turn an off-screen curve into artificial horizontal or vertical
        lines along the image edge. OpenCV clips line segments during drawing.
        """

        if path_points is None or len(path_points) < 2:
            return

        points = np.asarray(
            path_points,
            dtype=np.float64,
        )

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]

        if len(points) < 2:
            return

        h, w = overlay.shape[:2]

        # Prevent unsafe integer conversion for pathological values while
        # keeping the safety limits well outside the visible frame.
        points[:, 0] = np.clip(
            points[:, 0],
            -4.0 * w,
            5.0 * w,
        )
        points[:, 1] = np.clip(
            points[:, 1],
            -4.0 * h,
            5.0 * h,
        )

        points = np.rint(points).astype(np.int32)
        points = points.reshape(-1, 1, 2)

        cv2.polylines(
            overlay,
            [points],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_obstacle_boxes(
        self,
        overlay: np.ndarray,
        mask: np.ndarray,
        min_area: int = MIN_OBSTACLE_AREA_PX,
    ) -> None:
        """Draw bounding boxes around vehicle-class components."""

        obstacle_mask = (
            mask == CLASS_VEHICLE
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            obstacle_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                self.obstacle_box,
                2,
            )

    def draw_ego_axis(
        self,
        overlay: np.ndarray,
        ego_center_x: float,
    ) -> None:
        """Draw the dotted robot-center axis."""

        h, w = overlay.shape[:2]

        ego_center = int(
            np.clip(
                round(ego_center_x),
                0,
                w - 1,
            )
        )

        for y in range(0, h, 6):
            cv2.circle(
                overlay,
                (ego_center, y),
                1,
                self.ego_axis,
                -1,
            )

    def driving_overlay(
        self,
        mask: np.ndarray,
        analyzer: LaneAnalyzer,
        lane_width_px: float,
        camera_offset_x_px: float,
        roi_ratios: Tuple[float, float],
    ) -> np.ndarray:
        """
        Run lane analysis and draw the detected paths.

        lane_width_px and roi_ratios remain in this method signature for
        compatibility with the existing callers.
        """

        _, w = mask.shape

        (
            yellow_fit,
            white_fit,
            center_path,
            _,
            _,
            _,
            _,
        ) = analyzer.analyze(mask)

        overlay = self.colorize(mask)

        if yellow_fit is not None:
            self.draw_path(
                overlay,
                yellow_fit.path_points,
                self.curve_yellow,
                2,
            )

        if white_fit is not None:
            self.draw_path(
                overlay,
                white_fit.path_points,
                self.curve_white,
                2,
            )

        self.draw_path(
            overlay,
            center_path,
            self.curve_center,
            3,
        )

        self.draw_obstacle_boxes(
            overlay,
            mask,
        )

        ego_center = float(
            w / 2.0 + camera_offset_x_px
        )

        self.draw_ego_axis(
            overlay,
            ego_center,
        )

        return overlay