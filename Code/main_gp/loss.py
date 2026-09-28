# loss.py
import math

import torch
import torch.nn.functional as F

# ── Loss weights ───────────────────────────────────────────────
# till best setup lambda_smoothl1 0.2, lambda_fft 1, lambda_pearson 0
LAMBDA_PEARSON   = 0 # best 1
LAMBDA_SMOOTHL1  = 0.5#0.2 #0,5 better with fft 1 #better 0.9 with pearson 1 # good 0.8  with pearson 1
LAMBDA_FFT       = 1  #0.7

# HR band used by spectral losses (Hz)
HR_LOW_HZ  = 0.75   # 45 BPM
HR_HIGH_HZ = 2.50   # 150 BPM


# ── Signal utilities ───────────────────────────────────────────

def center_signal(y: torch.Tensor) -> torch.Tensor:
    """
    y: [B, T]
    Subtract per-sample mean along time axis.
    """
    return y - y.mean(dim=-1, keepdim=True)


def _hann_window(T: int, device: torch.device) -> torch.Tensor:
    """
    Returns a Hann window of length T on the given device.
    Applied before FFT to reduce spectral leakage.
    """
    return torch.hann_window(T, periodic=False, device=device)


def _rfft_psd(signal: torch.Tensor) -> torch.Tensor:
    """
    signal: [B, T], assumed zero-mean
    Returns power spectral density: [B, T//2+1]
    Applies Hann window internally.
    """
    B, T = signal.shape
    win  = _hann_window(T, signal.device)         # [T]
    psd  = torch.fft.rfft(signal * win, dim=-1)   # [B, T//2+1] complex
    return psd.abs() ** 2                          # [B, T//2+1] real power


def _freq_axis(T: int, fs: float, device: torch.device) -> torch.Tensor:
    """
    Returns the frequency axis (Hz) for rfft output of length T
    sampled at fs Hz. Shape: [T//2+1]
    """
    return torch.fft.rfftfreq(T, d=1.0 / fs, device=device)





# ── Individual loss functions ──────────────────────────────────
def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    pred, target: [B, T], already zero-meaned recommended but not required.
    Returns scalar: 1 - mean Pearson r across batch.
    Range [0, 2]. Lower is better.
    Amplitude-invariant — only waveform shape and phase matter.
    """
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    pred_std = torch.sqrt((pred   * pred  ).sum(dim=-1) + eps)
    targ_std = torch.sqrt((target * target).sum(dim=-1) + eps)
    corr     = (pred * target).sum(dim=-1) / (pred_std * targ_std + eps)

    return (1.0 - corr).mean()





def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, T]
    Plain MSE on waveform amplitudes.
    Amplitude-sensitive — use only as a weak regularizer, not primary loss.
    """
    return F.mse_loss(pred, target)





def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, T]
    Plain MAE on waveform amplitudes.
    Slightly more robust to outlier frames than MSE.
    Same amplitude-sensitivity caveat as mse_loss.
    """
    return F.l1_loss(pred, target)






# def fft_loss(pred: torch.Tensor, target: torch.Tensor,
#              fs: float = 30.0) -> torch.Tensor:
#     """
#     pred, target: [B, T], zero-mean recommended.
#     fs          : sampling rate in Hz (assumed uniform across batch).

#     Computes MSE between the power spectra of pred and target,
#     restricted to the physiological cardiac band [HR_LOW_HZ, HR_HIGH_HZ].

#     This pushes the predicted spectrum to match the ground-truth
#     spectrum shape — complementary to NegPearson which only sees time domain.
#     """
#     T     = pred.shape[-1]
#     freqs = _freq_axis(T, fs, pred.device)            # [T//2+1]
#     mask  = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)  # [T//2+1] bool

#     psd_pred   = _rfft_psd(pred)   [:, mask]          # [B, n_bins]
#     psd_target = _rfft_psd(target) [:, mask]          # [B, n_bins]

#     # Normalize each PSD to sum=1 so amplitude differences don't dominate
#     psd_pred   = psd_pred   / (psd_pred.sum(dim=-1, keepdim=True)   + 1e-8)
#     psd_target = psd_target / (psd_target.sum(dim=-1, keepdim=True) + 1e-8)

#     return F.mse_loss(psd_pred, psd_target)



def fft_loss(pred, target, fs=30.0, pad_factor=8):
    # Zero-pad before FFT to recover fine frequency resolution
    # T=60 padded to 480 → resolution = 30/480 = 0.0625 Hz ≈ 0.4 bpm
    T = pred.shape[-1]
    N = T * pad_factor          # 60 × 8 = 480 points

    pred_f  = torch.fft.rfft(pred,   n=N, dim=-1).abs()
    target_f = torch.fft.rfft(target, n=N, dim=-1).abs()

    freqs = torch.fft.rfftfreq(N, d=1.0/fs, device=pred.device)
    mask  = (freqs >= 0.7) & (freqs <= 4.0)

    return F.l1_loss(pred_f[:, mask], target_f[:, mask])








def snr_loss(pred: torch.Tensor, hr_hz: torch.Tensor,
             fs: float = 30.0, bw: float = 0.1) -> torch.Tensor:
    """
    pred  : [B, T], zero-mean recommended.
    hr_hz : [B]    — ground-truth heart rate in Hz (bpm / 60).
    fs    : sampling rate in Hz.
    bw    : half-bandwidth of the signal window around GT HR (Hz).

    Maximizes SNR = signal_power / noise_power inside the cardiac band,
    where signal_power is the PSD integrated in [hr_hz - bw, hr_hz + bw]
    and noise_power is the rest of the cardiac band.

    Returns -SNR_linear (minimizing this maximizes SNR).
    """
    T     = pred.shape[-1]
    freqs = _freq_axis(T, fs, pred.device)            # [F]
    psd   = _rfft_psd(pred)                           # [B, F]

    # Cardiac band mask — shared for all samples
    band_mask = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)  # [F]

    # Signal window mask — per-sample, centered on GT HR
    hr_exp    = hr_hz.unsqueeze(-1)                   # [B, 1]
    freqs_exp = freqs.unsqueeze(0)                    # [1, F]
    sig_mask  = (
        (freqs_exp >= hr_exp - bw) &
        (freqs_exp <= hr_exp + bw) &
        band_mask.unsqueeze(0)
    )                                                 # [B, F]
    noise_mask = band_mask.unsqueeze(0) & ~sig_mask   # [B, F]

    sig_power   = (psd * sig_mask  ).sum(dim=-1)      # [B]
    noise_power = (psd * noise_mask).sum(dim=-1).clamp(min=1e-8)

    snr = sig_power / noise_power                     # [B], linear ratio
    return -snr.mean()                                # minimize -> maximize SNR





def hr_mae_loss(pred: torch.Tensor, hr_hz_gt: torch.Tensor,
                fs: float = 30.0) -> torch.Tensor:
    """
    pred     : [B, T], zero-mean recommended.
    hr_hz_gt : [B]   — ground-truth HR in Hz.
    fs       : sampling rate in Hz.

    Non-differentiable HR-MAE for monitoring only (argmax is not differentiable).
    Returns mean absolute error in BPM as a plain Python float.
    Call this with torch.no_grad() — do NOT use in .backward().

    For a rough differentiable approximation during training, use snr_loss instead.
    """
    T     = pred.shape[-1]
    freqs = _freq_axis(T, fs, pred.device)
    mask  = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)

    psd   = _rfft_psd(pred)[:, mask]                  # [B, n_bins]
    f_sub = freqs[mask]                               # [n_bins]

    # Peak frequency per sample
    peak_idx  = psd.argmax(dim=-1)                    # [B]
    pred_hz   = f_sub[peak_idx]                       # [B]

    mae_bpm = ((pred_hz - hr_hz_gt).abs() * 60.0).mean()
    return mae_bpm



# ── Spectral projection helpers (ported from external_loss.py) ─

# def _bpm_range(fmin_hz: float = 0.75, fmax_hz: float = 2.5,
#                device: torch.device = None) -> torch.Tensor:
#     """
#     Integer BPM bins covering the cardiac band.
#     Default: 45–150 BPM  (0.75–2.5 Hz).
#     """
#     bpm_min = int(round(fmin_hz * 60.0))
#     bpm_max = int(round(fmax_hz * 60.0))
#     return torch.arange(bpm_min, bpm_max + 1, device=device)



def _bpm_range(fmin_hz: float = 0.67, fmax_hz: float = 3.0,
               device: torch.device = None) -> torch.Tensor:
    """
    Integer BPM bins covering the cardiac band.
    Default: 40–180 BPM  (0.67–3.0 Hz) — full physiological range.
    """
    bpm_min = int(round(fmin_hz * 60.0))
    bpm_max = int(round(fmax_hz * 60.0))
    return torch.arange(bpm_min, bpm_max + 1, device=device)




def _spectral_energy(signal: torch.Tensor, fs: float,
                     bpm_bins: torch.Tensor) -> torch.Tensor:
    """
    signal   : [T]  — single sample, zero-mean recommended.
    fs       : sampling rate in Hz.
    bpm_bins : [K]  — integer BPM values to evaluate.

    Projects the windowed signal onto sinusoids at each BPM bin
    and returns a normalized energy distribution over [K] bins.

    This is the same sinusoidal-projection approach used in the
    RhythmMamba / RhythmFormer toolbox — avoids torch.fft.rfft
    bin-alignment issues when T is short or fs is non-integer.
    """
    T      = signal.numel()
    device = signal.device
    dtype  = signal.dtype

    # Hann window — reduces spectral leakage
    win    = torch.hann_window(T, device=device, dtype=dtype)
    x_win  = signal * win                                  # [T]

    # Convert BPM to fractional DFT bin index k = freq_hz * T / fs
    freq_hz = bpm_bins.to(device=device, dtype=dtype) / 60.0  # [K]
    k       = freq_hz * T / fs                                 # [K]

    # Phase matrix [K, T]
    t     = torch.arange(T, device=device, dtype=dtype)
    phase = k.unsqueeze(1) * (2.0 * math.pi * t / T).unsqueeze(0)  # [K, T]

    # Project onto sin and cos basis — equivalent to |DFT[k]|²
    sin_e = (x_win.unsqueeze(0) * torch.sin(phase)).sum(dim=1)  # [K]
    cos_e = (x_win.unsqueeze(0) * torch.cos(phase)).sum(dim=1)  # [K]
    energy = sin_e ** 2 + cos_e ** 2                             # [K]

    # Normalize to a probability distribution over BPM bins
    return energy / (energy.sum() + 1e-12)                       # [K]


# ═══════════════════════════════════════════════════════════════
#  FFT-based spectral energy  (Prof. Jang's directive)
#
#  P(f) = |FFT(ŷ(t))|²   with zero-pad to n_fft (default 16384)
#
#  Pipeline:
#    ŷ(t) → Hann window → zero-pad to n_fft → torch.fft.rfft
#         → |·|² → extract power at target BPM bins → normalize
#
#  Resolution at n_fft=16384, fs=30 Hz:
#    Δf = fs / n_fft = 30 / 16384 ≈ 0.00183 Hz ≈ 0.11 BPM
#
#  For each target BPM value, the nearest FFT bin is found via:
#    bin_index = round( (BPM / 60) × n_fft / fs )
#
#  Maximum bin-alignment error: 0.11 / 2 ≈ 0.055 BPM — negligible
#  for all practical HR estimation purposes.
#
#  Fully differentiable: torch.fft.rfft supports autograd, and the
#  bin indices are fixed (computed from GT, not from the predicted
#  signal), so gradients flow through the PSD values back to the
#  predicted waveform.
# ═══════════════════════════════════════════════════════════════

def _spectral_energy_fft(signal: torch.Tensor, fs: float,
                         bpm_bins: torch.Tensor,
                         n_fft: int = 16384) -> torch.Tensor:
    """
    FFT-based spectral energy at specified BPM bins.

    signal   : [T]  — single sample, zero-mean recommended.
    fs       : sampling rate in Hz.
    bpm_bins : [K]  — integer BPM values to evaluate.
    n_fft    : zero-pad length (default 16384 per Prof. Jang's directive).

    Returns a normalized energy distribution over [K] bins, analogous
    to _spectral_energy() but using zero-padded FFT instead of
    sinusoidal projection.

    Resolution comparison (fs=30, T=60):
      _spectral_energy      : evaluates at exact BPM values, 1 BPM spacing
      _spectral_energy_fft  : FFT grid at 0.11 BPM spacing, nearest-bin lookup

    Both return [K]-shaped normalized distributions suitable for
    concentration_loss and harmonic_rank_loss.
    """
    T      = signal.numel()
    device = signal.device
    dtype  = signal.dtype

    # ── Step 1: Hann window on the original signal ────────────
    win   = torch.hann_window(T, device=device, dtype=dtype)
    x_win = signal * win                                       # [T]

    # ── Step 2: Zero-pad to n_fft and compute rfft ────────────
    # torch.fft.rfft with n > T implicitly zero-pads.
    # This increases FFT grid density without adding real information,
    # but the 0.11 BPM bin spacing eliminates bin-alignment error
    # when reading power at specific target frequencies.
    spectrum = torch.fft.rfft(x_win, n=n_fft, dim=-1)          # [n_fft//2+1] complex
    psd      = spectrum.abs() ** 2                              # [n_fft//2+1] real power

    # ── Step 3: Map target BPM values to nearest FFT bin indices ──
    freq_hz     = bpm_bins.to(device=device, dtype=dtype) / 60.0   # [K] Hz
    bin_indices = (freq_hz * n_fft / fs).round().long()             # [K] integer indices

    # Clamp to valid rfft output range [0, n_fft//2]
    max_bin     = n_fft // 2
    bin_indices = bin_indices.clamp(0, max_bin)

    # ── Step 4: Extract power at target bins ──────────────────
    # Indexing with fixed integer indices is differentiable w.r.t. psd values.
    # The indices depend on bpm_bins and fs (constants), not on the signal.
    energy = psd[bin_indices]                                      # [K]

    # ── Step 5: Normalize to a probability distribution ───────
    # Same normalization as _spectral_energy: sum over the HR band = 1.
    # This makes C_GT = sum of energy in the neighborhood window,
    # consistent with both methods.
    return energy / (energy.sum() + 1e-12)                         # [K]


def _peak_bpm_index(signal: torch.Tensor, fs: float,
                    bpm_bins: torch.Tensor) -> int:
    """
    Returns the index into bpm_bins with the highest spectral energy.
    Used to find GT HR bin from the ground-truth BVP waveform.
    """
    return int(_spectral_energy(signal, fs, bpm_bins).argmax().item())



def _gaussian_dist(n_bins: int, center: int, std: float,
                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Normalized Gaussian distribution over n_bins, peaked at center.
    Used as a soft target for the KL divergence HR loss.
    std controls how many BPM bins of tolerance are allowed.
    """
    idx  = torch.arange(n_bins, device=device, dtype=dtype)
    dist = torch.exp(-((idx - float(center)) ** 2) / (2.0 * std ** 2))
    dist = dist.clamp(min=1e-15)
    return dist / dist.sum()


# ── CE loss (RhythmMamba) ──────────────────────────────────────

def frequency_ce_loss(pred: torch.Tensor, target: torch.Tensor,
                      fs: torch.Tensor | float) -> torch.Tensor:
    """
    pred   : [B, T]
    target : [B, T]  — ground-truth BVP waveform
    fs     : scalar float or [B] tensor — sampling rate in Hz

    Treats the normalized spectral energy of pred as a class probability
    distribution over BPM bins, and the dominant BPM bin of target as
    the class label.  Cross-entropy then pushes the predicted spectrum
    to peak at the correct HR bin.

    Gradient flows through the sinusoidal projection of pred —
    the GT bin index from target is treated as a fixed label (no gradient).

    From: RhythmMamba  loss = 0.2 * NegPearson + 1.0 * FreqCE
    """
    B      = pred.shape[0]
    bins   = _bpm_range(device=pred.device)    # [K]
    losses = []

    for b in range(B):
        fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)

        # GT bin — argmax on target spectrum, no gradient needed
        gt_idx = _peak_bpm_index(target[b], fs_b, bins)

        # Predicted spectrum as log-probability — [1, K] for cross_entropy
        pred_spec = _spectral_energy(pred[b], fs_b, bins).unsqueeze(0)   # [1, K]
        gt_label  = torch.tensor([gt_idx], device=pred.device)           # [1]

        # F.cross_entropy expects raw logits or log-probs
        # pred_spec is already a positive distribution, so log it
        losses.append(F.cross_entropy(torch.log(pred_spec + 1e-12), gt_label))

    return torch.stack(losses).mean()


# ── KL HR distribution loss (RhythmFormer) ────────────────────
def hr_kl_loss(pred: torch.Tensor, target: torch.Tensor,
               fs: "torch.Tensor | float",
               std: float = 3.0) -> torch.Tensor:
    """
    pred   : [B, T]
    target : [B, T]  — ground-truth BVP waveform
    fs     : scalar float or [B] tensor — sampling rate in Hz
    std    : Gaussian std in BPM bins (tolerance around GT peak).
             std=1 → ±1 BPM tolerance, std=3 → default, std=5 → permissive.

    Minimises KL( pred_spectral_distribution || GT_Gaussian ) per batch item.

    The predicted distribution is the sinusoidal-projection spectral energy
    of pred — fully differentiable, gradient flows back to all 3 Conv1d weights.
    The GT distribution is a soft Gaussian centred at the GT waveform's
    dominant BPM — treated as a fixed label, no gradient required.

    Learnable parameters added: 0
    """
    from loss import _bpm_range, _spectral_energy, _peak_bpm_index, _gaussian_dist

    B      = pred.shape[0]
    bins   = _bpm_range(device=pred.device)   # [K]  40–180 BPM
    K      = bins.numel()
    losses = []

    for b in range(B):
        fs_b = (
            float(fs[b].item())
            if torch.is_tensor(fs) and fs.dim() > 0
            else float(fs)
        )

        # ── GT Gaussian target — non-differentiable argmax is fine here ───────
        gt_idx  = _peak_bpm_index(target[b], fs_b, bins)                   # int
        gt_dist = _gaussian_dist(K, gt_idx, std, pred.device, pred.dtype)  # [K] fixed

        # ── Predicted distribution — MUST be differentiable w.r.t. pred ──────
        # _spectral_energy uses sin/cos inner products, no argmax, no .item()
        pred_spec = _spectral_energy(pred[b], fs_b, bins)   # [K]  gradient flows

        # ── KL( pred_spec || gt_dist ): push pred spectrum toward GT Gaussian ─
        losses.append(
            F.kl_div(torch.log(pred_spec + 1e-12), gt_dist, reduction="sum")
        )

    return torch.stack(losses).mean()






import math

def hilbert_freq_loss(pred, target, fs=30.0):
    """
    Continuous instantaneous frequency loss via Hilbert transform.
    Bypasses FFT bin-width limit — gives exact Hz, not binned.
    Both pred and target: [B, T]
    """
    def analytic(x):
        # Hilbert transform via FFT (one-sided spectrum doubling)
        Xf = torch.fft.fft(x, dim=-1)
        N  = x.shape[-1]
        h  = torch.zeros(N, device=x.device, dtype=Xf.dtype)
        if N % 2 == 0:
            h[0] = h[N//2] = 1
            h[1:N//2] = 2
        else:
            h[0] = 1
            h[1:(N+1)//2] = 2
        return torch.fft.ifft(Xf * h, dim=-1)

    def unwrap(phi):
        # PyTorch-safe phase unwrap along last dim
        d = torch.diff(phi, dim=-1)
        d_mod = (d + math.pi) % (2 * math.pi) - math.pi
        d_mod = torch.where((d_mod == -math.pi) & (d > 0),
                            torch.tensor(math.pi, device=phi.device), d_mod)
        correction = d_mod - d
        correction = torch.where(d.abs() < math.pi,
                                 torch.zeros_like(correction), correction)
        return phi + F.pad(correction.cumsum(dim=-1), (1, 0))

    pred_a   = analytic(pred)
    target_a = analytic(target)

    inst_f_pred   = torch.diff(unwrap(torch.angle(pred_a)),   dim=-1) * fs / (2 * math.pi)
    inst_f_target = torch.diff(unwrap(torch.angle(target_a)), dim=-1) * fs / (2 * math.pi)

    # only compare within plausible BVP band (0.7–4 Hz) to ignore noise regions
    mask = (inst_f_target >= 0.7) & (inst_f_target <= 4.0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return F.l1_loss(inst_f_pred[mask], inst_f_target[mask])





def subharmonic_loss(pred: torch.Tensor, target: torch.Tensor,
                     fs: "torch.Tensor | float",
                     margin: float = 0.05) -> torch.Tensor:
    """
    Penalizes spectral energy at HALF the true HR (the sub-harmonic / octave trap).

    For each sample, finds the GT BPM bin, then enforces that the predicted
    energy at the FUNDAMENTAL exceeds the energy at GT/2 by a margin:

        loss = relu( energy(GT/2) - energy(GT) + margin )

    Fully differentiable (reuses _spectral_energy — no argmax on pred).
    Adds 0 learnable parameters. Does not touch inference.
    Only fires when GT/2 falls inside the 40-180 BPM band.
    """
    B    = pred.shape[0]
    bins = _bpm_range(device=pred.device)          # 40..180 BPM
    K    = bins.numel()
    bpm_min = int(bins[0].item())                  # 40
    losses = []

    for b in range(B):
        fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)

        # GT fundamental bin (non-diff label is fine)
        gt_idx = _peak_bpm_index(target[b], fs_b, bins)
        gt_bpm = bpm_min + gt_idx

        # predicted distribution — differentiable
        pred_spec = _spectral_energy(pred[b], fs_b, bins)   # [K]

        half_idx = int(round(gt_bpm / 2.0)) - bpm_min
        if 0 <= half_idx < K:
            # energy at fundamental (small window) vs at sub-harmonic
            e_fund = pred_spec[max(gt_idx-1,0):gt_idx+2].sum()
            e_half = pred_spec[max(half_idx-1,0):half_idx+2].sum()
            losses.append(F.relu(e_half - e_fund + margin))
        else:
            losses.append(torch.tensor(0.0, device=pred.device))

    return torch.stack(losses).mean()



def peaksharp_loss(pred, target, fs, margin=0.10):
    B = pred.shape[0]
    bins = _bpm_range(device=pred.device)
    losses = []

    for b in range(B):
        fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)

        gt_idx = _peak_bpm_index(target[b], fs_b, bins)
        pred_spec = _spectral_energy(pred[b], fs_b, bins)

        gt_power = pred_spec[gt_idx]

        wrong_spec = pred_spec.clone()
        left = max(0, gt_idx - 1)
        right = min(len(pred_spec), gt_idx + 2)
        wrong_spec[left:right] = 0.0

        wrong_peak = wrong_spec.max()

        losses.append(F.relu(wrong_peak - gt_power + margin))

    return torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════
#  STAGE A — Spectral concentration loss  (Prof. Jang, step 2)
# ═══════════════════════════════════════════════════════════════
#
#  L_total          = L_Pearson + λ · L_concentration
#  L_concentration  = − log( C_GT + ε )
#
#              Σ P(f)  for f in N(f_GT)          (power near the true HR)
#  C_GT   =  ─────────────────────────────
#              Σ P(f)  for f in B_HR    + ε      (total power in the HR band)
#
#  P(f)   = |FFT(ŷ(t))|²                          (predicted power spectrum)
#  N(f_GT)= [f_GT − δ, f_GT + δ]                  (narrow neighborhood of true HR)
#  B_HR   = physiological HR frequency band       (whole cardiac band)
#
#  Two P(f) engines are available, selectable via the `method` parameter:
#
#    method="sinusoidal" (default):
#      _spectral_energy() — sinusoidal projection at exact integer BPM values.
#      Resolution: 1 BPM, zero bin-alignment error.
#      The normalized distribution sums to 1 over B_HR, so C_GT reduces to
#      the summed energy inside the neighborhood window.
#
#    method="fft":
#      _spectral_energy_fft() — zero-padded FFT per Prof. Jang's directive.
#      Resolution: ~0.11 BPM at n_fft=16384, fs=30.
#      Same normalization convention: output sums to 1, C_GT is the
#      windowed sum.
#
#  Both methods are fully differentiable through the predicted waveform.
# ═══════════════════════════════════════════════════════════════

def concentration_loss(pred: torch.Tensor, target: torch.Tensor,
                       fs: "torch.Tensor | float",
                       delta_bpm: int = 3,
                       eps: float = 1e-8,
                       method: str = "sinusoidal",
                       n_fft: int = 16384) -> torch.Tensor:
    """
    Stage A spectral concentration loss.

    pred      : [B, T]  — predicted BVP waveform.
    target    : [B, T]  — ground-truth BVP waveform (supplies f_GT).
    fs        : scalar float or [B] tensor — sampling rate in Hz.
    delta_bpm : half-width of the neighborhood N(f_GT) in BPM bins.
                delta_bpm=3 covers f_GT ± 3 BPM.
    eps       : stabilizer inside the log and the ratio.
    method    : "sinusoidal" — evaluate P(f) via sin/cos projection at
                               exact integer BPM values (1 BPM resolution).
                "fft"        — evaluate P(f) via zero-padded FFT at nearest
                               FFT bins (0.11 BPM resolution at n_fft=16384).
    n_fft     : FFT length when method="fft". Ignored for method="sinusoidal".

    Returns  mean over batch of  − log( C_GT + eps ).

    C_GT is the fraction of the predicted spectrum's power that falls in a
    narrow window around the true HR, relative to the total power in the HR
    band.  Minimizing − log(C_GT) drives all predicted spectral energy to
    concentrate at the true frequency, so any leakage to a harmonic or
    sub-harmonic lowers C_GT and is penalized.

    Fully differentiable through both spectral energy methods.
    Adds 0 learnable parameters and does not affect inference.
    """
    if method not in ("sinusoidal", "fft"):
        raise ValueError(f"Unknown spectral method '{method}'. "
                         f"Expected 'sinusoidal' or 'fft'.")

    B    = pred.shape[0]
    bins = _bpm_range(device=pred.device)          # [K]  40..180 BPM
    K    = bins.numel()
    losses = []

    for b in range(B):
        fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)

        # GT fundamental bin from the target waveform (fixed label, no gradient).
        # Always uses sinusoidal projection for GT — the method choice only
        # affects how the *predicted* spectrum is computed for gradient flow.
        gt_idx = _peak_bpm_index(target[b], fs_b, bins)

        # Predicted normalized power distribution over the HR band (sum = 1).
        if method == "fft":
            pred_spec = _spectral_energy_fft(pred[b], fs_b, bins, n_fft=n_fft)
        else:
            pred_spec = _spectral_energy(pred[b], fs_b, bins)

        lo = max(gt_idx - delta_bpm, 0)
        hi = min(gt_idx + delta_bpm + 1, K)
        c_gt = pred_spec[lo:hi].sum()               # scalar in [0, 1]

        losses.append(-torch.log(c_gt + eps))

    return torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════
#  STAGE B — Harmonic / sub-harmonic ranking losses
# ═══════════════════════════════════════════════════════════════
#
#  L_sub  = max( 0,  m + P(0.5·f_GT) − P(f_GT) )
#  L_harm = max( 0,  m + P(2·f_GT)   − P(f_GT) )
#
#  Margin-based hinge losses that force the fundamental peak P(f_GT) to exceed
#  the sub-harmonic peak P(0.5·f_GT) and the harmonic peak P(2·f_GT) by at
#  least the margin m.  Zero when the fundamental already dominates by m;
#  positive (penalizing) when a false peak comes within m of the fundamental.
#
#  Escalation step: use only if Pearson + concentration still leaves frequent
#  half-frequency or double-frequency errors.
#
#  Supports both P(f) engines via the `method` parameter,
#  identical to concentration_loss.
# ═══════════════════════════════════════════════════════════════

def harmonic_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                       fs: "torch.Tensor | float",
                       margin: float = 0.10,
                       delta_bpm: int = 3,
                       use_sub: bool = True,
                       use_harm: bool = True,
                       method: str = "sinusoidal",
                       n_fft: int = 16384) -> torch.Tensor:
    """
    Stage B harmonic / sub-harmonic ranking loss.

    pred      : [B, T]  — predicted BVP waveform.
    target    : [B, T]  — ground-truth BVP waveform (supplies f_GT).
    fs        : scalar float or [B] tensor — sampling rate in Hz.
    margin    : hinge margin m — how much taller the fundamental peak must be
                than each false peak (in normalized-power units).
    delta_bpm : half-width (BPM bins) of the small window summed around each
                peak location, giving robustness to exact bin placement.
    use_sub   : include L_sub (penalizes the 0.5·f_GT sub-harmonic).
    use_harm  : include L_harm (penalizes the 2·f_GT harmonic).
    method    : "sinusoidal" — evaluate P(f) via sin/cos projection at
                               exact integer BPM values (1 BPM resolution).
                "fft"        — evaluate P(f) via zero-padded FFT at nearest
                               FFT bins (0.11 BPM resolution at n_fft=16384).
    n_fft     : FFT length when method="fft". Ignored for method="sinusoidal".

    Returns the mean over batch of the summed active hinge terms.

    L_sub  = relu( m + P(0.5·f_GT) − P(f_GT) )
    L_harm = relu( m + P(2·f_GT)   − P(f_GT) )

    Each false-peak term is only counted when its target bin (0.5·f_GT or
    2·f_GT) falls inside the 40-180 BPM band; otherwise that term is skipped
    for that sample.  Fully differentiable through both spectral energy methods.
    """
    if method not in ("sinusoidal", "fft"):
        raise ValueError(f"Unknown spectral method '{method}'. "
                         f"Expected 'sinusoidal' or 'fft'.")

    B    = pred.shape[0]
    bins = _bpm_range(device=pred.device)          # [K]  40..180 BPM
    K    = bins.numel()
    bpm_min = int(bins[0].item())                  # 40
    losses = []

    def _window_power(spec: torch.Tensor, center_idx: int) -> torch.Tensor:
        lo = max(center_idx - delta_bpm, 0)
        hi = min(center_idx + delta_bpm + 1, K)
        return spec[lo:hi].sum()

    for b in range(B):
        fs_b = float(fs[b].item()) if torch.is_tensor(fs) and fs.dim() > 0 else float(fs)

        # GT fundamental bin (fixed label, no gradient)
        gt_idx = _peak_bpm_index(target[b], fs_b, bins)
        gt_bpm = bpm_min + gt_idx

        # Predicted spectrum — differentiable, method-selectable
        if method == "fft":
            pred_spec = _spectral_energy_fft(pred[b], fs_b, bins, n_fft=n_fft)
        else:
            pred_spec = _spectral_energy(pred[b], fs_b, bins)

        p_fund = _window_power(pred_spec, gt_idx)

        term = torch.tensor(0.0, device=pred.device)

        if use_sub:
            sub_idx = int(round(gt_bpm / 2.0)) - bpm_min
            if 0 <= sub_idx < K:
                p_sub = _window_power(pred_spec, sub_idx)
                term = term + F.relu(margin + p_sub - p_fund)

        if use_harm:
            harm_idx = int(round(gt_bpm * 2.0)) - bpm_min
            if 0 <= harm_idx < K:
                p_harm = _window_power(pred_spec, harm_idx)
                term = term + F.relu(margin + p_harm - p_fund)

        losses.append(term)

    return torch.stack(losses).mean()




def compute_loss(pred, target, fs=30.0, hr_hz=None, weights: dict = None, n_fft: int = 16384):
    """
    Flexible loss combinator. Pass a dict of {loss_name: weight}.

    Available loss names:
        "pearson"       — neg_pearson_loss
        "fft"           — fft_loss
        "smoothl1"      — smooth_l1_loss
        "snr"           — snr_loss         (requires hr_hz)
        "hilbert"       — hilbert_freq_loss  (continuous instantaneous frequency)
        "ce"            — frequency_ce_loss
        "kl"            — hr_kl_loss
        "subharm"       — subharmonic_loss
        "peaksharp"     — peaksharp_loss
        "concentration" — concentration_loss   (Stage A: spectral concentration)
        "harmrank"      — harmonic_rank_loss    (Stage B: harmonic ranking)

    spectral_method : controls how P(f) is computed inside concentration_loss
                      and harmonic_rank_loss. Does not affect other losses.
        "sinusoidal" — sin/cos projection at exact integer BPM (default).
        "fft"        — zero-padded FFT per Prof. Jang's directive.

    n_fft : FFT length when spectral_method="fft" (default 16384).
            Ignored when spectral_method="sinusoidal".

    Example combinations:
        # Stage A — Pearson + concentration (sinusoidal, default)
        weights = {"pearson": 1.0, "concentration": 1.0}

        # Stage A — Pearson + concentration 
        weights = {"pearson": 1.0, "concentration": 1.0}
        spectral_method = "fft"

        # Stage B — add explicit harmonic ranking on top of Stage A
        weights = {"pearson": 1.0, "concentration": 1.0, "harmrank": 0.5}

        # Other combinations
        weights = {"fft": 1.0, "pearson": 1.0}
        weights = {"hilbert": 1.0}
    """
    
    spectral_method: str = "fft"

    if weights is None:
        #weights = {"ce": 1.0, "pearson": 0.2}
        #weights = {"ce": 1.0,"pearson": 0.2,"peaksharp": 0.5,}
        #weights = {"hilbert": 1.0}
        #weights = {"ce": 1.0, "pearson": 0.2, "subharm": 0.5}
        #weights = {"ce": 1.0,"pearson": 0.2,"peaksharp": 0.5,}
        #weights = {"pearson": 1.0}
        # ── Stage A  — Pearson + concentration ──
        weights = {"pearson": 1.0, "concentration": 1.0}
        # ── Stage B — add harmonic ranking only if half/double errors persist ──
        #weights = {"pearson": 1.0, "concentration": 1.0, "harmrank": 0.5}

    p = center_signal(pred)
    t = center_signal(target)

    loss_fns = {
        "pearson":       lambda: neg_pearson_loss(p, t),
        "fft":           lambda: fft_loss(p, t, fs=fs),
        "smoothl1":      lambda: F.smooth_l1_loss(p, t),
        "snr":           lambda: snr_loss(p, hr_hz, fs=fs),
        "hilbert":       lambda: hilbert_freq_loss(p, t, fs=fs),
        "ce":            lambda: frequency_ce_loss(p, t, fs=fs),
        "kl":            lambda: hr_kl_loss(p, t, fs=fs),
        "subharm":       lambda: subharmonic_loss(p, t, fs=fs),
        "peaksharp":     lambda: peaksharp_loss(p, t, fs=fs),
        "concentration": lambda: concentration_loss(p, t, fs=fs,
                                                    method=spectral_method,
                                                    n_fft=n_fft,delta_bpm=3),       # Stage A
        "harmrank":      lambda: harmonic_rank_loss(p, t, fs=fs,
                                                    method=spectral_method,
                                                    n_fft=n_fft),       # Stage B
    }

    # validate snr requirement
    if "snr" in weights and hr_hz is None:
        raise ValueError("snr loss requires hr_hz")

    stats = {k: 0.0 for k in ["loss_total", "pearson", "fft",
                              "smoothl1", "snr", "hilbert", "ce", "kl",
                              "subharm", "peaksharp",
                              "concentration", "harmrank"]}

    loss = torch.tensor(0.0, device=pred.device)
    for name, w in weights.items():
        if w == 0.0:
            continue
        l = loss_fns[name]()
        loss = loss + w * l
        stats[name] = float(l.detach().cpu())

    stats["loss_total"] = float(loss.detach().cpu())

    # remap to match training engine meter keys
    return loss, {
        "loss_total":        stats["loss_total"],
        "loss_pearson":      stats["pearson"],
        "loss_smoothl1":     stats["smoothl1"],
        "loss_fft":          stats["fft"],
        "loss_snr":          stats["snr"],
        "loss_hilbert":      stats["hilbert"],
        "loss_ce":           stats["ce"],
        "loss_kl":           stats["kl"],
        "loss_subharm":      stats["subharm"],
        "loss_peaksharp":    stats["peaksharp"],
        "loss_concentration": stats["concentration"],
        "loss_harmrank":     stats["harmrank"],
    }
