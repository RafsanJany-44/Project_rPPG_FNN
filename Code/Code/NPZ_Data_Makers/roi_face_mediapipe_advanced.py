# roi_face_mediapipe_advanced.py
"""
Advanced face ROI using MediaPipe Face Mesh.

We build free-form ROIs from facial landmarks:
- We define sets of landmarks and shape types (polygon, circle, rect_between).
- For each frame, we detect the face mesh, convert landmarks to pixels,
  and rasterize the selected shapes into a binary mask.
- We then apply this mask to the input frame and return the masked ROI image.

We keep the design flexible so we can change the ROI definition by editing
the ROI_SHAPES list below.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from pathlib import Path


try:
    import mediapipe as mp
except ImportError as e:
    mp = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# -------------------------------------------------------------------
# 1. ROI SHAPE CONFIGURATION
#    We can edit this list to change which regions we use.
# -------------------------------------------------------------------
ROI_SHAPES: List[Dict[str, Any]] = [
    
    {
        "name": "forehead_strip",
        "type": "polygon",
        # Approximate forehead band using FaceMesh indices
        "indices": [21, 54, 103, 67, 109, 10, 338, 297, 332, 284,298, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 71],
    },
    {
        "name": "left_cheek_big",   # viewer's left cheek (subject's right)
        "type": "polygon",
        "indices": [143, 116, 123, 187, 207, 203, 209, 198, 217, 174, 188, 245, 233, 232, 231, 230, 229, 228, 31, 35],
    },
    {
        "name": "right_cheek_big",  # viewer's right cheek (subject's left)
        "type": "polygon",
        "indices": [372, 345, 352, 411, 427, 423, 429, 420, 437, 399, 412, 465, 453, 452, 451, 450, 449, 448, 261, 265],
    },

]


class MediaPipeFaceMeshRoi:
    """
    Advanced ROI based on MediaPipe FaceMesh and free-form shapes.

    Parameters
    ----------
    roi_shapes : list of dict
        Configuration of regions. Each dict can define:
          - type="polygon", with "indices": [list of landmark indices]
          - type="circle",  with "center_idx" and "radius_px"
          - type="rect_between", with "idx1", "idx2", "x_margin", "y_frac"
    refine_landmarks, min_detection_confidence, min_tracking_confidence :
        Standard MediaPipe FaceMesh options.

    Methods
    -------
    extract(img_bgr) -> roi_img
        Standard ROI interface: returns masked image.
    extract_with_mask(img_bgr) -> (roi_img, mask, debug_shapes)
        Extended interface for debugging and visualization.
    """

    def __init__(
        self,
        roi_shapes: List[Dict[str, Any]] = ROI_SHAPES,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "mediapipe is required for MediaPipeFaceMeshRoi but "
                "could not be imported. Install it with `pip install mediapipe`."
            ) from _IMPORT_ERROR

        self.roi_shapes = roi_shapes

        self.mp_face_mesh = mp.solutions.face_mesh
        # We use static_image_mode=True so each frame is treated independently.
        # If we want tracking, we can set static_image_mode=False later.
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # --------------------------------------------------------------
    # Internal helpers: landmarks and shapes
    # --------------------------------------------------------------
    def _landmarks_to_xy(
        self,
        landmarks,
        img_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        We convert the 3D normalized landmarks to 2D pixel coordinates.

        Returns
        -------
        pts : np.ndarray of shape (468, 2)
            Each row is (x, y) in pixel coordinates.
        """
        h, w, _ = img_shape
        xs = []
        ys = []
        for lm in landmarks:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        pts = np.stack([xs, ys], axis=1)
        return pts

    def _mask_polygon(
        self,
        h: int,
        w: int,
        polygon_pts: np.ndarray,
    ) -> np.ndarray:
        """
        We rasterize a polygon into a binary mask.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        if polygon_pts.shape[0] < 3:
            return mask

        pts_int = np.round(polygon_pts).astype(np.int32)
        pts_int = pts_int.reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts_int], 1)
        return mask

    def _mask_circle(
        self,
        h: int,
        w: int,
        center_xy: Tuple[float, float],
        radius_px: float,
    ) -> np.ndarray:
        """
        We rasterize a circle into a binary mask.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = center_xy
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        rad_i = int(round(radius_px))
        cv2.circle(mask, (cx_i, cy_i), rad_i, 1, thickness=-1)
        return mask

    def _mask_rect_between(
        self,
        h: int,
        w: int,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        x_margin: float,
        y_frac: float,
    ) -> np.ndarray:
        """
        We build a rectangle using two landmarks as anchors.

        We take the vertical span between pt1 and pt2, scale it by y_frac,
        center it around the midpoint, and expand horizontally by x_margin.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1 = pt1
        x2, y2 = pt2

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        dy = abs(y2 - y1)

        rect_height = dy * y_frac
        rect_width = dy * (1.0 + x_margin)

        x_min = int(round(cx - rect_width / 2.0))
        x_max = int(round(cx + rect_width / 2.0))
        y_min = int(round(cy - rect_height / 2.0))
        y_max = int(round(cy + rect_height / 2.0))

        x_min = max(0, min(w, x_min))
        x_max = max(0, min(w, x_max))
        y_min = max(0, min(h, y_min))
        y_max = max(0, min(h, y_max))

        if x_max <= x_min or y_max <= y_min:
            return mask

        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
        return mask

    # --------------------------------------------------------------
    # Core mask construction
    # --------------------------------------------------------------
    def _build_roi_mask_and_shapes(
        self,
        img_bgr: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        We build a combined ROI mask from all shapes in self.roi_shapes.

        Returns
        -------
        mask : (H, W) uint8 array with values 0 or 1
        debug_shapes : list of polygon point arrays for visualization
        """
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        debug_shapes: List[np.ndarray] = []

        for cfg in self.roi_shapes:
            shape_type = cfg.get("type", "polygon")

            if shape_type == "polygon":
                indices = cfg.get("indices", [])
                if not indices:
                    continue
                poly_pts = pts[indices, :]  # (N, 2)
                m = self._mask_polygon(h, w, poly_pts)
                mask |= m
                debug_shapes.append(poly_pts)

            elif shape_type == "circle":
                center_idx = cfg.get("center_idx", None)
                radius_px = cfg.get("radius_px", None)
                if center_idx is None or radius_px is None:
                    continue
                center_xy = tuple(pts[center_idx, :])
                m = self._mask_circle(h, w, center_xy, radius_px)
                mask |= m
                # We approximate circle as a small polygon for debug display
                theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                cx, cy = center_xy
                circle_pts = np.stack(
                    [cx + radius_px * np.cos(theta),
                     cy + radius_px * np.sin(theta)],
                    axis=1,
                )
                debug_shapes.append(circle_pts)

            elif shape_type == "rect_between":
                idx1 = cfg.get("idx1", None)
                idx2 = cfg.get("idx2", None)
                x_margin = cfg.get("x_margin", 0.0)
                y_frac = cfg.get("y_frac", 1.0)
                if idx1 is None or idx2 is None:
                    continue
                pt1 = tuple(pts[idx1, :])
                pt2 = tuple(pts[idx2, :])
                m = self._mask_rect_between(h, w, pt1, pt2, x_margin, y_frac)
                mask |= m
                # For debug, we can store the rectangle corners as a polygon
                ys, xs = np.where(m > 0)
                if ys.size > 0 and xs.size > 0:
                    x_min = xs.min()
                    x_max = xs.max()
                    y_min = ys.min()
                    y_max = ys.max()
                    rect_poly = np.array(
                        [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                        ],
                        dtype=np.float32,
                    )
                    debug_shapes.append(rect_poly)

        return mask, debug_shapes
    
    def _build_masks_by_name(
        self,
        img_bgr: np.ndarray,
        pts: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Build one binary mask per ROI name (0/1 uint8), instead of a single union mask.
        """
        h, w = img_bgr.shape[:2]
        masks: Dict[str, np.ndarray] = {}

        for cfg in self.roi_shapes:
            name = cfg.get("name", None)
            if not name:
                continue

            shape_type = cfg.get("type", "polygon")
            m = np.zeros((h, w), dtype=np.uint8)

            if shape_type == "polygon":
                indices = cfg.get("indices", [])
                if indices:
                    poly_pts = pts[indices, :]
                    m = self._mask_polygon(h, w, poly_pts)

            elif shape_type == "circle":
                center_idx = cfg.get("center_idx", None)
                radius_px = cfg.get("radius_px", None)
                if center_idx is not None and radius_px is not None:
                    center_xy = tuple(pts[center_idx, :])
                    m = self._mask_circle(h, w, center_xy, radius_px)

            elif shape_type == "rect_between":
                idx1 = cfg.get("idx1", None)
                idx2 = cfg.get("idx2", None)
                x_margin = cfg.get("x_margin", 0.0)
                y_frac = cfg.get("y_frac", 1.0)
                if idx1 is not None and idx2 is not None:
                    pt1 = tuple(pts[idx1, :])
                    pt2 = tuple(pts[idx2, :])
                    m = self._mask_rect_between(h, w, pt1, pt2, x_margin, y_frac)

            # If same name appears multiple times, union them
            if name in masks:
                masks[name] |= m
            else:
                masks[name] = m

        return masks

    # --------------------------------------------------------------
    # Extract masks by name
    # --------------------------------------------------------------
    def extract_masks_by_name(
        self,
        img_bgr: np.ndarray,
        fallback_full_image_on_fail: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Return dict {roi_name: mask} for the configured ROI_SHAPES.
        Masks are uint8 with values 0/1.

        If face detection fails:
          - if fallback_full_image_on_fail=True: return full mask for all ROIs
          - else: return empty masks for all ROIs
        """
        h, w = img_bgr.shape[:2]

        img_rgb = img_bgr[:, :, ::-1]
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            if fallback_full_image_on_fail:
                return {cfg["name"]: np.ones((h, w), dtype=np.uint8) for cfg in self.roi_shapes if "name" in cfg}
            return {cfg["name"]: np.zeros((h, w), dtype=np.uint8) for cfg in self.roi_shapes if "name" in cfg}

        face_landmarks = results.multi_face_landmarks[0].landmark
        pts = self._landmarks_to_xy(face_landmarks, img_bgr.shape)

        return self._build_masks_by_name(img_bgr, pts)


    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def extract_with_mask(
        self,
        img_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        We compute the ROI for one frame and also return the mask and shapes.

        Returns
        -------
        roi_img : (H, W, 3) uint8
            Input image with everything outside ROI set to zero.
        mask : (H, W) uint8
            Binary mask (0 outside ROI, 1 inside ROI).
        debug_shapes : list of np.ndarray
            Each array is a set of polygon points in (x, y) for visualization.
        """
        h, w = img_bgr.shape[:2]

        # MediaPipe expects RGB
        img_rgb = img_bgr[:, :, ::-1]

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            # We fall back to whole image if detection fails
            mask = np.ones((h, w), dtype=np.uint8)
            roi_img = img_bgr.copy()
            return roi_img, mask, []

        face_landmarks = results.multi_face_landmarks[0].landmark
        pts = self._landmarks_to_xy(face_landmarks, img_bgr.shape)

        mask, debug_shapes = self._build_roi_mask_and_shapes(img_bgr, pts)

        # If mask is empty, we fall back to full face bounding box
        if np.count_nonzero(mask) == 0:
            x_min = int(np.floor(np.min(pts[:, 0])))
            x_max = int(np.ceil(np.max(pts[:, 0])))
            y_min = int(np.floor(np.min(pts[:, 1])))
            y_max = int(np.ceil(np.max(pts[:, 1])))
            x_min = max(0, min(w, x_min))
            x_max = max(0, min(w, x_max))
            y_min = max(0, min(h, y_min))
            y_max = max(0, min(h, y_max))
            mask = np.zeros((h, w), dtype=np.uint8)
            if x_max > x_min and y_max > y_min:
                mask[y_min:y_max, x_min:x_max] = 1
                rect_poly = np.array(
                    [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    dtype=np.float32,
                )
                debug_shapes.append(rect_poly)

        roi_img = img_bgr.copy()
        roi_img[mask == 0] = 0

        return roi_img, mask, debug_shapes

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Standard RoiExtractor interface: we only return the masked image.
        """
        roi_img, _, _ = self.extract_with_mask(img_bgr)
        return roi_img
