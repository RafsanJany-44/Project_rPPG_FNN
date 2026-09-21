# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
import math
import numpy as np
import torch

# HR evaluation band
BPM_MIN = 42.0
BPM_MAX = 240.0




def normalize_input_per_window(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, T, 3]
    Per-sample, per-channel normalization across time.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def select_roi_rgb(x: torch.Tensor, roi_index):
    """
    x: [B, T, 9]

    roi_index:
        0, 1, 2  -> single ROI
        "avg"    -> average RGB of all 3 ROIs

    returns: [B, T, 3]
    """
    if roi_index == "avg":
        B, T, C = x.shape

        if C % 3 != 0:
            raise ValueError(f"C={C} is not divisible by 3.")

        n_rois = C // 3

        # [B, T, 9] -> [B, T, 3 ROIs, 3 RGB]
        x = x.view(B, T, n_rois, 3)

        # average over ROI dimension
        return x.mean(dim=2)   # [B, T, 3]

    else:
        roi_index = int(roi_index)
        start = roi_index * 3
        end = start + 3

        if x.size(-1) < end:
            raise ValueError(
                f"Input has C={x.size(-1)}, but ROI_INDEX={roi_index} needs [{start}:{end}]"
            )

        return x[..., start:end]







def fft_peak_bpm_1d(sig: np.ndarray, fs: float, bpm_min: float = 42.0, bpm_max: float = 240.0) -> float:
    """
    Very simple FFT peak BPM estimate from a 1D signal.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 1 or len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")

    sig = sig - np.mean(sig)
    n = len(sig)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(sig)) ** 2

    fmin = bpm_min / 60.0
    fmax = bpm_max / 60.0
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return float("nan")

    peak_f = freqs[mask][np.argmax(spec[mask])]
    return float(peak_f * 60.0)




# def fft_peak_bpm_1d(sig: np.ndarray, fs: float, 
#                     bpm_min: float = 42.0, bpm_max: float = 240.0,
#                     pad_factor: int = 4) -> float:
#     sig = np.asarray(sig, dtype=np.float64).reshape(-1)
#     if len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
#         return float("nan")

#     sig = sig - np.mean(sig)
#     n = len(sig)
#     N = n * pad_factor                          # ADD

#     freqs = np.fft.rfftfreq(N, d=1.0 / fs)      # n → N
#     spec = np.abs(np.fft.rfft(sig, n=N)) ** 2   # add n=N

#     fmin = bpm_min / 60.0
#     fmax = bpm_max / 60.0
#     mask = (freqs >= fmin) & (freqs <= fmax)

#     if not np.any(mask):
#         return float("nan")

#     peak_f = freqs[mask][np.argmax(spec[mask])]
#     return float(peak_f * 60.0)




@torch.no_grad()
def batch_hr_mae(pred: torch.Tensor, target: torch.Tensor, fs: torch.Tensor) -> float:
    """
    pred, target: [B, T]
    fs: [B]
    """
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()
    fs_np = fs.detach().cpu().numpy()

    maes = []
    for i in range(pred_np.shape[0]):
        bpm_p = fft_peak_bpm_1d(pred_np[i], float(fs_np[i]), BPM_MIN, BPM_MAX)
        bpm_t = fft_peak_bpm_1d(targ_np[i], float(fs_np[i]), BPM_MIN, BPM_MAX)
        if np.isfinite(bpm_p) and np.isfinite(bpm_t):
            maes.append(abs(bpm_p - bpm_t))

    if len(maes) == 0:
        return float("nan")
    return float(np.mean(maes))


def format_metrics(prefix: str, metrics: dict) -> str:
    parts = [f"{prefix}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isnan(v):
                parts.append(f"{k}=nan")
            else:
                parts.append(f"{k}={v:.5f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
