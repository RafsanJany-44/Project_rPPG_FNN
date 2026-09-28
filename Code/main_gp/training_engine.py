#training_engine.py
import math
import random
from helper import (
    select_roi_rgb,
    normalize_input_per_window,
    fft_peak_bpm_1d,
    batch_hr_mae,
    format_metrics
)
from tqdm import tqdm



from loss import center_signal, compute_loss
import torch
import torch.nn.functional as F

# def seed_everything(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

#     torch.use_deterministic_algorithms(True, warn_only=True)

# seed_everything(42)


# def get_aug_prob(epoch: int) -> float:
#     if epoch < 10:
#         return 0.3              # 30% chance each augmentation applies
#     elif epoch < 30:
#         return 0.6              # 60% chance
#     else:
#         return 0.8




def get_loss_weights(epoch):
    if epoch < 20:
        # Phase 1: fft anchors HR roughly
        return {"fft": 1.0, "snr": 0.5}
    elif epoch < 50:
        # Phase 2: introduce hilbert gently
        return {"fft": 0.5, "snr": 0.5, "hilbert": 0.5}
    else:
        # Phase 3: hilbert dominates fine-tuning
        return {"fft": 0.2, "snr": 0.3, "hilbert": 1.0}






import torch
@torch.no_grad()
def _estimate_bpm_from_y(
    y: torch.Tensor,
    fs: float,
    bpm_min: float = 42.0,
    bpm_max: float = 175.0,
    pad_factor: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate dominant BPM from one GT BVP window using FFT peak."""
    y = y.float()
    T = y.numel()
    if T < 8 or fs <= 1.0:
        return torch.tensor(float("nan"), device=y.device)

    y = y - y.mean()
    win = torch.hann_window(T, periodic=False, device=y.device, dtype=y.dtype)
    N = int(T * pad_factor)

    psd = torch.fft.rfft(y * win, n=N).abs().pow(2)
    freqs_bpm = torch.fft.rfftfreq(N, d=1.0 / float(fs), device=y.device) * 60.0

    mask = (freqs_bpm >= bpm_min) & (freqs_bpm <= bpm_max)
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=y.device)

    psd_band = psd[mask]
    freqs_band = freqs_bpm[mask]
    if psd_band.sum() <= eps:
        return torch.tensor(float("nan"), device=y.device)

    return freqs_band[torch.argmax(psd_band)]


@torch.no_grad()
def frequency_scale_temporal_aug(
    X: torch.Tensor,
    Y: torch.Tensor,
    fs: torch.Tensor,
    training: bool = True,
    aug_scale: torch.Tensor | None = None,
    p_temporal: float = 1.0,
    n: float = 0.10,
    m: float = 0.10,
    bpm_min_safe: float = 42.0,
    bpm_max_safe: float = 175.0,
    p_time_flip: float = 0.5,
    real_scale_tol: float = 1e-6,
):
    """
    Offline-aware, HR-safe temporal augmentation.

    Behavior:
      1) Real/original samples, aug_scale == 1.0:
            - NO online frequency scaling
            - optional time flip only

      2) Offline augmented samples, aug_scale != 1.0:
            - desired final scale neighborhood: [A - n, A + m]
            - convert to online multiplier: [(A - n)/A, (A + m)/A]
            - estimate current HR from Y
            - clip online multiplier so final HR stays in [bpm_min_safe, bpm_max_safe]
            - if the clipped interval is valid, warp X and Y together
            - optional time flip

    Important:
      X and Y are always warped/flipped together to preserve RGB-GT alignment.

    Args:
        X: [B, T, 3]
        Y: [B, T]
        fs: [B]
        aug_scale: [B], parsed from manifest seq/meta.
                   real sample -> 1.0, e.g. 01-01
                   offline aug -> factor, e.g. 1.4 for 01-01_x1.4
    """
    if not training:
        return X, Y, fs

    B, T, C = X.shape
    device = X.device

    if aug_scale is None:
        aug_scale = torch.ones(B, device=device, dtype=torch.float32)
    else:
        aug_scale = aug_scale.to(device=device, dtype=torch.float32).view(-1)
        if aug_scale.numel() != B:
            raise ValueError(f"aug_scale must have shape [B]. Got {tuple(aug_scale.shape)} for B={B}.")

    X_aug = X.clone()
    Y_aug = Y.clone()
    base_t = torch.arange(T, device=device).float()

    for b in range(B):
        A = float(aug_scale[b].item())

        is_real = abs(A - 1.0) <= real_scale_tol
        do_temporal = (not is_real) and (torch.rand(1, device=device).item() < p_temporal)
        #do_temporal = (torch.rand(1) < p_temporal)                                            #did not work

        if do_temporal and A > 0.0:
            target_low = max(A - float(n), 1e-6)
            target_high = max(A + float(m), target_low)

            desired_low = target_low / A
            desired_high = target_high / A

            fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)
            current_bpm = _estimate_bpm_from_y(
                Y_aug[b],
                fs=fs_b,
                bpm_min=bpm_min_safe,
                bpm_max=bpm_max_safe,
            )

            if torch.isfinite(current_bpm):
                current_bpm_f = float(current_bpm.item())
                safe_low = bpm_min_safe / max(current_bpm_f, 1e-6)
                safe_high = bpm_max_safe / max(current_bpm_f, 1e-6)

                final_low = max(desired_low, safe_low)
                final_high = min(desired_high, safe_high)

                if final_low < final_high:
                    scale = torch.empty(1, device=device).uniform_(final_low, final_high).item()

                    pos = (base_t * scale) % T
                    idx0 = torch.floor(pos).long()
                    idx1 = (idx0 + 1) % T
                    w = pos - idx0.float()

                    x0 = X_aug[b, idx0, :]
                    x1 = X_aug[b, idx1, :]
                    X_aug[b] = (1.0 - w).unsqueeze(-1) * x0 + w.unsqueeze(-1) * x1

                    y0 = Y_aug[b, idx0]
                    y1 = Y_aug[b, idx1]
                    Y_aug[b] = (1.0 - w) * y0 + w * y1

        if torch.rand(1, device=device).item() < p_time_flip:
            X_aug[b] = torch.flip(X_aug[b], dims=[0])
            Y_aug[b] = torch.flip(Y_aug[b], dims=[0])

    return X_aug, Y_aug, fs


@torch.no_grad()
def single_channel_rgb_gain_drop_aug(
    X: torch.Tensor,
    training: bool = True,
    p: float = 0.10,
    gain_range=(1.10, 1.25),
    drop_range=(0.80, 0.90),
    p_gain: float = 0.5,
):
    """
    Simple RGB amplitude augmentation.

    Behavior:
      - Applies per sample with probability p.
      - Randomly chooses exactly ONE channel: R, G, or B.
      - Randomly chooses gain or drop.
      - Scales only the AC component of the chosen channel.
      - Leaves the other two channels untouched.
      - Does NOT modify Y because RGB amplitude change does not change HR timing.

    X: [B, T, 3]
    """
    if not training:
        return X

    B, T, C = X.shape
    assert C == 3, f"Expected X shape [B,T,3], got {X.shape}"

    device = X.device
    apply_mask = torch.rand(B, device=device) < p
    if apply_mask.sum() == 0:
        return X

    X_aug = X.clone()
    mean = X_aug.mean(dim=1, keepdim=True)
    ac = X_aug - mean
    ac_out = ac.clone()

    channels = torch.randint(low=0, high=3, size=(B,), device=device)
    use_gain = torch.rand(B, device=device) < p_gain

    gain_scale = torch.empty(B, device=device).uniform_(*gain_range)
    drop_scale = torch.empty(B, device=device).uniform_(*drop_range)
    scale = torch.where(use_gain, gain_scale, drop_scale)

    for c in range(3):
        mask_c = apply_mask & (channels == c)
        if mask_c.any():
            ac_out[mask_c, :, c] = ac[mask_c, :, c] * scale[mask_c].view(-1, 1)

    X_aug[apply_mask] = mean[apply_mask] + ac_out[apply_mask]
    return X_aug




@torch.no_grad()
def single_channel_rgb_dc_shift_aug(
    x: torch.Tensor,
    training: bool = True,
    p: float = 0.05,
    scale_range=(0.95, 1.05),
):
    """
    Very mild RGB brightness/color-baseline augmentation.

    Per window:
      - randomly choose exactly one channel: R/G/B
      - multiply the whole selected channel
      - Y is NOT changed

    x: [B, T, 3]
    """
    if not training or p <= 0:
        return x

    B, T, C = x.shape
    assert C == 3, f"Expected x shape [B,T,3], got {x.shape}"

    device = x.device
    out = x.clone()

    apply_mask = torch.rand(B, device=device) < p
    if apply_mask.sum() == 0:
        return out

    chosen_channel = torch.randint(low=0, high=3, size=(B,), device=device)
    scale = torch.empty(B, device=device).uniform_(*scale_range)

    for b in range(B):
        if not apply_mask[b]:
            continue

        c = int(chosen_channel[b].item())
        out[b, :, c] = out[b, :, c] * scale[b]

    return out






def get_hr_hz(bvp_gt: torch.Tensor, fs: float = 30.0) -> torch.Tensor:  # only nned when using the snr loss, otherwise not needed
    # bvp_gt: [B, T]
    freqs = torch.fft.rfftfreq(bvp_gt.shape[-1], d=1.0/fs, device=bvp_gt.device)
    psd   = torch.fft.rfft(bvp_gt, dim=-1).abs() ** 2
    mask  = (freqs >= 0.75) & (freqs <= 2.5)
    psd_masked       = psd.clone()
    psd_masked[:, ~mask] = 0
    peak_idx = psd_masked.argmax(dim=-1)          # [B]
    return freqs[peak_idx]                        # [B], Hz




import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    low_hz: float = 0.67,
    high_hz: float = 3.0,
    order: int = 3,
) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    
    if len(sig) < (order * 3 + 1):
        return sig.copy()
    
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    
    if not (0 < low < high < 1):
        return sig.copy()
    
    b, a = butter(order, [low, high], btype="band")
    
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig.copy()



def apply_bandpass_batch(pred, fs):
    """
    Apply bandpass filter to batch.
    Returns filtered tensor that maintains gradient connection.
    """
    B, T = pred.shape
    filtered = []
    
    for i in range(B):
        # Keep on same device, convert to numpy only for filtering
        p = pred[i].detach().cpu().numpy()
        p_filt = bandpass_filter(p, fs=fs, low_hz=0.67, high_hz=3.0)
        filtered.append(torch.from_numpy(p_filt.copy()).float())
    
    # Stack and move to device
    filtered = torch.stack(filtered).to(pred.device)
    
    # CRITICAL: This creates new tensor without gradients
    # We need to "attach" it back to the computation graph
    # Use a trick: multiply by 1.0 from original pred
    
    # Create a mask that connects filtered output to original pred
    alpha = torch.zeros_like(pred).requires_grad_(False) + 1.0
    filtered = filtered + alpha * (pred - pred.detach())
    
    return filtered







def run_one_epoch(model, loader, optimizer=None, DEVICE=None, ROI_INDEX=None, epoch=None):
    is_train = optimizer is not None
    model.train(is_train)


    meter = {
        "loss_total":    0.0,
        "loss_pearson":  0.0,
        "loss_smoothl1": 0.0,
        "loss_fft":      0.0,
        "loss_snr":      0.0,
        "loss_hilbert":  0.0, 
        "hr_mae":        0.0,
        "loss_ce":       0.0,
        "loss_kl":       0.0,
        "loss_subharm":  0.0,
        "loss_peaksharp": 0.0,
        "n":             0,
    }

    for batch in tqdm(loader, desc="Running epoch"):
        if len(batch) == 5:
            X, Y, t, fs, aug_scale = batch
        elif len(batch) == 4:
            X, Y, t, fs = batch
            aug_scale = torch.ones(X.size(0), dtype=torch.float32)
        else:
            raise ValueError(f"Expected batch of length 4 or 5, got {len(batch)}")

        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        fs = fs.to(DEVICE, non_blocking=True)
        aug_scale = aug_scale.to(DEVICE, non_blocking=True).float()


        X_roi = select_roi_rgb(X, ROI_INDEX)          # [B,T,3]


        X_roi, Y, fs = frequency_scale_temporal_aug(
            X_roi,
            Y,
            fs,
            training=is_train,
            aug_scale=aug_scale,
            p_temporal=1.0,
            n=0.20,
            m=0.20,
            bpm_min_safe=42.0,
            bpm_max_safe=175.0,
            p_time_flip=0,              # no time flip for now, since we are using FFT-based loss
        )


        X_roi = single_channel_rgb_gain_drop_aug(
            X_roi,
            training=is_train,
            p=0.30,
            gain_range=(1.10, 1.25),
            drop_range=(0.80, 0.90),
            p_gain=0.5,
        )


        pred = model(X_roi)                           # [B,T]

    
        # safe length match
        T = min(pred.size(-1), Y.size(-1))
        pred = pred[:, :T]
        Y = Y[:, :T]

        hr_hz = get_hr_hz(Y, fs=fs.mean().item())  # [Hz], only needed when using the snr loss, otherwise not needed
        #loss, stats = compute_total_loss(pred, Y)

        #loss, stats = compute_loss(pred, Y, fs=fs.mean().item(), hr_hz=hr_hz)  # for debugging individual loss components
        #weights = get_loss_weights(epoch)
        #loss, stats = compute_loss(pred, Y, fs=fs.mean().item(),hr_hz=hr_hz, weights=weights)
        #loss, stats = compute_loss(pred, Y, fs=fs.mean().item(),hr_hz=hr_hz)
        loss, stats = compute_loss(pred, Y, fs=fs,hr_hz=hr_hz)
        #loss = loss +  con_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        hr_mae = batch_hr_mae(center_signal(pred), center_signal(Y), fs)

        bs = X.size(0)
        meter["loss_total"]    += stats["loss_total"]    * bs
        meter["loss_pearson"]  += stats["loss_pearson"]  * bs
        meter["loss_smoothl1"] += stats["loss_smoothl1"] * bs
        meter["loss_fft"]      += stats["loss_fft"]      * bs
        meter["loss_snr"]      += stats["loss_snr"]      * bs
        meter["loss_hilbert"]  += stats["loss_hilbert"]  * bs 
        meter["hr_mae"]        += (0.0 if math.isnan(hr_mae) else hr_mae) * bs
        meter["loss_ce"]       += stats["loss_ce"] * bs
        meter["loss_kl"]       += stats["loss_kl"] * bs
        meter["loss_subharm"]  += stats["loss_subharm"] * bs
        meter["loss_peaksharp"] += stats["loss_peaksharp"] * bs
        meter["n"]             += bs

    n = max(1, meter["n"])
    return {k: meter[k] / n for k in meter if k != "n"}
