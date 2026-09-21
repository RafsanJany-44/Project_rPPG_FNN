# main/rgb_extractor_multi_roi.py
#
# Our shared multi-ROI extractor for PURE / UBFC / UBFC-Phys.
#
# We KEEP the existing RGB-only function for backward compatibility:
#   extract_rgb_timeseries_multi_roi_from_video(...)
#
# We ADD a new feature extractor (Option A++):
#   extract_feat_timeseries_multi_roi_from_video(...)
#
# Feature channels per ROI (F = 8):
#   [R, G, B, valid, pix_frac, meanY, stdY, motionY]
# => total channels C = F*K
#
# Notes:
# - If frame_times_s is provided (PURE): we use it, and stop when frames end.
# - If frame_times_s is None (UBFC/UBFC-Phys): we build t = i/fps from video metadata.
# - motionY is computed inside ROI as mean absolute frame-to-frame delta of Y (luma),
#   so it is causal (uses previous frame).
#
# We/our: first person plural perspective is used in comments as requested.

from typing import List, Optional, Callable, Tuple
from pathlib import Path
import numpy as np
import cv2


# ----------------------------
# our pixel helpers
# ----------------------------
def _safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x is None or x.size == 0:
        return 0.0, 0.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(m):
        m = 0.0
    if not np.isfinite(s):
        s = 0.0
    return m, s


def _mean_rgb_robust_from_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rgb_low: int = 55,
    rgb_high: int = 200,
) -> np.ndarray:
    # We apply optional preprocessing first (e.g., CLAHE/unsharp).
    if preprocess_fn is not None:
        img_bgr = preprocess_fn(img_bgr)

    if mask01 is None or mask01.size == 0:
        return np.zeros(3, dtype=np.float32)

    m = mask01.astype(bool)
    if m.sum() < 10:
        return np.zeros(3, dtype=np.float32)

    roi_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    pix = roi_rgb[m].reshape(-1, 3)
    if pix.size == 0:
        return np.zeros(3, dtype=np.float32)

    # We use our robust validity (pyVHR-like) to drop near-black / near-white pixels.
    valid = ~(
        (pix[:, 0] <= rgb_low) &
        (pix[:, 1] <= rgb_low) &
        (pix[:, 2] <= rgb_low)
    ) & ~(
        (pix[:, 0] >= rgb_high) &
        (pix[:, 1] >= rgb_high) &
        (pix[:, 2] >= rgb_high)
    )

    #pix = pix[valid] # ---------------------------------------------------------------------------------------------------flage
    if pix.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)

    return pix.mean(axis=0).astype(np.float32)


def _roi_y_stats_and_motion(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    prev_y_roi: Optional[np.ndarray],
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[float, float, float, Optional[np.ndarray], int, float]:
    """
    Returns:
      meanY, stdY, motionY, y_roi (for next step), n_pix, pix_frac
    """
    if preprocess_fn is not None:
        img_bgr = preprocess_fn(img_bgr)

    if mask01 is None or mask01.size == 0:
        return 0.0, 0.0, 0.0, None, 0, 0.0

    m = mask01.astype(bool)
    n_pix = int(m.sum())
    if n_pix < 10:
        return 0.0, 0.0, 0.0, None, n_pix, 0.0

    h, w = img_bgr.shape[:2]
    denom = float(max(1, h * w))
    pix_frac = float(n_pix / denom)

    # We compute luma Y from RGB. Our input frame is BGR.
    # Convert to RGB then luma (same weights you used before):
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B  # [H,W]

    y_roi = Y[m].reshape(-1).astype(np.float32)  # [n_pix]
    meanY, stdY = _safe_mean_std(y_roi)

    # We compute motionY as mean absolute difference between current and previous ROI luma vectors.
    # If the mask size changes, we still compute motion in a stable way by matching lengths.
    motionY = 0.0
    if prev_y_roi is not None and prev_y_roi.size >= 10 and y_roi.size >= 10:
        n = int(min(prev_y_roi.size, y_roi.size))
        if n >= 10:
            motionY = float(np.mean(np.abs(y_roi[:n] - prev_y_roi[:n])))

    if not np.isfinite(motionY):
        motionY = 0.0

    return float(meanY), float(stdY), float(motionY), y_roi, n_pix, pix_frac


# ----------------------------
# our original RGB-only extractor (unchanged behavior)
# ----------------------------
def extract_rgb_timeseries_multi_roi_from_video(
    video_path,
    frame_times_s: Optional[np.ndarray],
    roi_extractor,
    roi_names: List[str],
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rgb_low: int = 55,
    rgb_high: int = 200,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_arr: [T]
      X:    [T, 3*K] in fixed roi_names order, channels are RGB per ROI.

    Notes:
      - If frame_times_s is provided (PURE): we use it, and stop when frames end.
      - If frame_times_s is None (UBFC/UBFC-Phys): we build t = i/fps from video metadata.
    """
    from tqdm import tqdm

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        n_total = 10**9

    if frame_times_s is not None:
        n_target = int(len(frame_times_s))
    else:
        n_target = int(n_total)

    K = len(roi_names)
    t_list = []
    X_list = []

    iterator = range(n_target)
    if show_progress:
        iterator = tqdm(iterator, desc=f"MultiROI RGB {Path(video_path).name}", ncols=100)

    for i in iterator:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        masks = roi_extractor.extract_masks_by_name(frame_bgr)
        row = np.zeros((3 * K,), dtype=np.float32)

        for k, name in enumerate(roi_names):
            m = masks.get(name, None)
            if m is None:
                rgb = np.zeros(3, dtype=np.float32)
            else:
                rgb = _mean_rgb_robust_from_mask(
                    frame_bgr,
                    m,
                    preprocess_fn=preprocess_fn,
                    rgb_low=rgb_low,
                    rgb_high=rgb_high,
                )
            row[k * 3:(k + 1) * 3] = rgb

        if frame_times_s is not None:
            t_list.append(float(frame_times_s[i]))
        else:
            t_list.append(float(i / fps))

        X_list.append(row)

    cap.release()

    t_arr = np.asarray(t_list, dtype=np.float64)
    X = np.asarray(X_list, dtype=np.float32)
    return t_arr, X


# ----------------------------
# our NEW feature extractor (Option A++ for ROI selection learning)
# ----------------------------
def extract_feat_timeseries_multi_roi_from_video(
    video_path,
    frame_times_s: Optional[np.ndarray],
    roi_extractor,
    roi_names: List[str],
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rgb_low: int = 55,
    rgb_high: int = 200,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_arr: [T]
      X:    [T, F*K] where F=8 channels per ROI:
            [R,G,B, valid, pix_frac, meanY, stdY, motionY] per ROI (roi_names order)
    """
    from tqdm import tqdm

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        n_total = 10**9

    if frame_times_s is not None:
        n_target = int(len(frame_times_s))
    else:
        n_target = int(n_total)

    K = len(roi_names)
    F = 8  # fixed
    t_list = []
    X_list = []

    # We keep previous ROI luma vectors to compute motionY.
    prev_y_roi = [None for _ in range(K)]

    iterator = range(n_target)
    if show_progress:
        iterator = tqdm(iterator, desc=f"MultiROI FEAT {Path(video_path).name}", ncols=100)

    for i in iterator:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        masks = roi_extractor.extract_masks_by_name(frame_bgr)

        row = np.zeros((F * K,), dtype=np.float32)

        for k, name in enumerate(roi_names):
            m = masks.get(name, None)

            # --- RGB robust mean ---
            rgb = np.zeros(3, dtype=np.float32)
            if m is not None:
                rgb = _mean_rgb_robust_from_mask(
                    frame_bgr,
                    m,
                    preprocess_fn=preprocess_fn,
                    rgb_low=rgb_low,
                    rgb_high=rgb_high,
                )

            # valid is 1 if RGB is not all zeros (our robustness already zeros invalid)
            valid = 1.0 if float(np.sum(np.abs(rgb))) > 0.0 else 0.0

            # --- ROI luma stats + motion ---
            meanY, stdY, motionY, y_roi_now, n_pix, pix_frac = _roi_y_stats_and_motion(
                frame_bgr,
                m,
                prev_y_roi=prev_y_roi[k],
                preprocess_fn=preprocess_fn,
            )
            prev_y_roi[k] = y_roi_now

            base = k * F
            row[base + 0] = float(rgb[0])
            row[base + 1] = float(rgb[1])
            row[base + 2] = float(rgb[2])
            row[base + 3] = float(valid)
            row[base + 4] = float(pix_frac)
            row[base + 5] = float(meanY)
            row[base + 6] = float(stdY)
            row[base + 7] = float(motionY)

        # timestamps
        if frame_times_s is not None:
            t_list.append(float(frame_times_s[i]))
        else:
            t_list.append(float(i / fps))

        X_list.append(row)

    cap.release()

    t_arr = np.asarray(t_list, dtype=np.float64)
    X = np.asarray(X_list, dtype=np.float32)
    return t_arr, X
