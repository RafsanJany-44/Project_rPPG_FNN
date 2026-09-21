# model_ZOO.py
# ═══════════════════════════════════════════════════════════════════════════════
# MEGA MODEL ZOO — Single source of truth for all rPPG model architectures.
#
# ALL models live here. Every training script imports from this ONE file.
#
# ORGANIZATION
# ────────────
#   SHARED        mean_normalize_rgb, TemporalBlock, UNet blocks
#   SECTION 1     Standalone projectors (non-TCN, non-UNet)
#   SECTION 2     Standalone TCNs (simple dilated + canonical Bai et al.)
#   SECTION 3     Standalone UNets (MiUNet, MMiUNet, MMMiUNet)
#   SECTION 4     Standalone frequency-domain models (FFN, BVPNet_V2)
#   SECTION 5     Three-branch fusion system (branches, fusions, ThreeBranchModel)
#   REGISTRY      All models registered for build_model(name)
#   SWEEP_ORDERS  Family-based sweep lists
#
# CAUTION
# ───────
#   Every class is an EXACT copy from its original source file.
#   No architecture has been modified. No forward pass has changed.
#   Adding out_ch class attributes to Branch backbones does NOT affect
#   nn.Parameter creation or seed-dependent initialization.
#
# Shared interface (all models):
#   input   x : (B, T, 3)     per-window RGB, raw
#   output  S : (B, T)        one 1-D pulse signal per window
#
# No second-person addressing in comments.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm
except Exception:
    from torch.nn.utils import weight_norm


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — RGB NORMALIZATION
# Source: model_ZOO_multi_branch.py (identical across all zoo files)
# ═══════════════════════════════════════════════════════════════════════════════

def mean_normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Per-window, per-channel mean normalization.
    Input:  x       [B, T, 3]
    Output: x_norm  [B, 3, T]   channels-first, ready for nn.Conv1d
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected input shape [B, T, 3], received {tuple(x.shape)}")
    c = x.transpose(1, 2)
    return c / (c.mean(dim=2, keepdim=True) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — CANONICAL TCN BUILDING BLOCKS (Bai et al. 2018)
# Source: model_ZOO_multi_branch.py (identical in model_ZOO_primitive.py)
#
# Two-conv-per-block residual TCN with weight-norm.
# Weights initialized from N(0, 0.01).
# Dilation doubles per block: d = 1, 2, 4, ...
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_amounts(kernel_size: int, dilation: int, causal: bool):
    """
    Total padding to preserve length = (k - 1) * d.
      causal     -> all on the left  (no future leakage)
      non-causal -> split symmetrically
    """
    pad = (kernel_size - 1) * dilation
    if causal:
        return pad, 0
    left = pad // 2
    return left, pad - left


class TemporalBlock(nn.Module):
    """
    One canonical TCN residual block:
        x -> [dilated conv -> weight-norm -> ReLU -> dropout]
          -> [dilated conv -> weight-norm -> ReLU -> dropout]
          -> (+) -> ReLU -> out
        (1x1 conv on the skip only when channel widths differ)
    """
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout, causal):
        super().__init__()
        self.pad_left, self.pad_right = _pad_amounts(kernel_size, dilation, causal)
        conv1 = nn.Conv1d(n_in, n_out, kernel_size, dilation=dilation)
        conv2 = nn.Conv1d(n_out, n_out, kernel_size, dilation=dilation)
        nn.init.normal_(conv1.weight, 0.0, 0.01)
        nn.init.normal_(conv2.weight, 0.0, 0.01)
        self.conv1 = weight_norm(conv1)
        self.conv2 = weight_norm(conv2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def _pad(self, h):
        return F.pad(h, (self.pad_left, self.pad_right))

    def forward(self, x):
        out = self.drop(self.relu(self.conv1(self._pad(x))))
        out = self.drop(self.relu(self.conv2(self._pad(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — U-NET BUILDING BLOCKS
# Source: model_ZOO_multi_branch.py (identical in model_ZOO_unet_family.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _unet_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
    )


def _unet_forward(enc1, enc2, enc3, pool, bottleneck,
                   up3, dec3, up2, dec2, up1, dec1, x):
    """Shared U-Net forward path. Returns dec1 output (before any final conv)."""
    T_in = x.size(-1)
    rem = T_in % 8
    if rem != 0:
        x = F.pad(x, (0, 8 - rem))
    e1 = enc1(x)
    e2 = enc2(pool(e1))
    e3 = enc3(pool(e2))
    b = bottleneck(pool(e3))
    d3 = dec3(torch.cat([up3(b), e3], dim=1))
    d2 = dec2(torch.cat([up2(d3), e2], dim=1))
    d1 = dec1(torch.cat([up1(d2), e1], dim=1))
    return d1[..., :T_in]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STANDALONE PROJECTORS (non-TCN, non-UNet)
# Source: model_ZOO_primitive.py
#
# Simple RGB-to-pulse projectors. No temporal depth beyond moving averages.
# These are the earliest experiment models and baselines.
# ═══════════════════════════════════════════════════════════════════════════════


# ── MODEL: TwoProjectionRGBProjector ────────────────────────────────────────
# Two learned RGB projections with adaptive alpha subtraction (CHROM-style).
#   Xs = projection 1,  Ys = projection 2
#   alpha = std(Xs) / std(Ys)
#   S = Xs - alpha * Ys
# Params: 6 (two 3-weight projections)
# RF: 1 sample (pointwise, but alpha uses whole-window std)
# ─────────────────────────────────────────────────────────────────────────────

class TwoProjectionRGBProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(
                [[[-0.07710419], [-0.57700866], [+0.19372641]],
                 [[+0.50026077], [+0.35686442], [-0.33997139]]],
                dtype=torch.float32,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        y = self.proj(x_norm)
        Xs = y[:, 0, :]
        Ys = y[:, 1, :]
        std_Xs = torch.std(Xs, dim=1, keepdim=True)
        std_Ys = torch.std(Ys, dim=1, keepdim=True)
        alpha = std_Xs / (std_Ys + 1e-8)
        return Xs - alpha * Ys


# ── MODEL: SimpleRGBProjector ───────────────────────────────────────────────
# Single pointwise RGB projection initialized at green channel.
# The simplest possible learned baseline.
# Params: 3
# RF: 1 sample
# ─────────────────────────────────────────────────────────────────────────────

class SimpleRGBProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor([[[0.0], [-1.0], [0.0]]], dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        return self.proj(x_norm).squeeze(1)

    def reach_samples(self) -> int:
        return 1


# ── MODEL: AdaptiveTrendProjector ───────────────────────────────────────────
# Green-channel projection + moving-average trend removal.
#   S0 = learned projection
#   trend = moving_average(S0)
#   lambda = std(trend) / std(S0)
#   S = S0 - lambda * trend
# Params: 3
# RF: 1 + trend_kernel (trend uses local context)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTrendProjector(nn.Module):
    def __init__(self, trend_kernel: int = 31):
        super().__init__()
        if trend_kernel < 3 or trend_kernel % 2 == 0:
            raise ValueError("trend_kernel must be an odd integer >= 3.")
        self.trend_kernel = int(trend_kernel)
        self.proj = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor([[[0.0], [-1.0], [0.0]]], dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        S0 = self.proj(x_norm).squeeze(1)
        pad = self.trend_kernel // 2
        S0_pad = F.pad(S0.unsqueeze(1), (pad, pad), mode="reflect")
        trend = F.avg_pool1d(S0_pad, kernel_size=self.trend_kernel, stride=1).squeeze(1)
        S0_c = S0 - S0.mean(dim=1, keepdim=True)
        trend_c = trend - trend.mean(dim=1, keepdim=True)
        std_S0 = torch.sqrt(torch.mean(S0_c ** 2, dim=1, keepdim=True) + 1e-8)
        std_trend = torch.sqrt(torch.mean(trend_c ** 2, dim=1, keepdim=True) + 1e-8)
        lam = std_trend / (std_S0 + 1e-8)
        lam = torch.clamp(lam, 0.0, 0.7)
        return S0 - lam * trend


# ── MODEL: TwoBranchAdaptive ────────────────────────────────────────────────
# Pulse branch + nuisance branch with high-pass cancellation.
#   X = a^T * RGB (pulse)
#   Y = b^T * RGB (nuisance), Yh = Y - moving_average(Y)
#   alpha = cov(X, Yh) / var(Yh)
#   S = X - alpha * Yh
# Params: 6
# RF: 1 + K (moving average kernel)
# ─────────────────────────────────────────────────────────────────────────────

class TwoBranchAdaptive(nn.Module):
    def __init__(self, K: int = 31):
        super().__init__()
        if K < 3 or K % 2 == 0:
            raise ValueError("K must be an odd integer >= 3.")
        self.K = int(K)
        self.a = nn.Parameter(torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = mean_normalize_rgb(x)
        X = (self.a.view(1, 3, 1) * c).sum(dim=1)
        Y = (self.b.view(1, 3, 1) * c).sum(dim=1)
        pad = self.K // 2
        Y_pad = F.pad(Y.unsqueeze(1), (pad, pad), mode="reflect")
        trend = F.avg_pool1d(Y_pad, kernel_size=self.K, stride=1).squeeze(1)
        Yh = Y - trend
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Yh - Yh.mean(dim=1, keepdim=True)
        alpha = (Xc * Yc).mean(dim=1, keepdim=True) / (Yc.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return X - alpha * Yh


# ── MODEL: AdaptiveDetrendProjector ─────────────────────────────────────────
# Two-stage projector:
#   Stage 1: two RGB projections + adaptive alpha cancellation (CHROM-style)
#   Stage 2: moving-average trend removal with one learnable gain gamma
# Params: 6 weights + 1 gamma = 7
# RF: 1 + trend_kernel
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDetrendProjector(nn.Module):
    def __init__(self, trend_kernel: int = 61, gamma_init: float = 0.5):
        super().__init__()
        if trend_kernel < 3 or trend_kernel % 2 == 0:
            raise ValueError("trend_kernel must be an odd integer >= 3.")
        self.trend_kernel = int(trend_kernel)
        self.proj = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, bias=False)
        g = min(max(float(gamma_init), 1e-4), 1.0 - 1e-4)
        self.gamma_raw = nn.Parameter(torch.log(torch.tensor(g / (1.0 - g))))
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(
                [[[-0.07710419], [-0.57700866], [+0.19372641]],
                 [[+0.50026077], [+0.35686442], [-0.33997139]]],
                dtype=torch.float32,
            ))

    def _moving_average(self, s: torch.Tensor) -> torch.Tensor:
        pad = self.trend_kernel // 2
        s_pad = F.pad(s.unsqueeze(1), (pad, pad), mode="reflect")
        return F.avg_pool1d(s_pad, kernel_size=self.trend_kernel, stride=1).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        y = self.proj(x_norm)
        Xs = y[:, 0, :]
        Ys = y[:, 1, :]
        alpha = torch.std(Xs, dim=1, keepdim=True) / (torch.std(Ys, dim=1, keepdim=True) + 1e-8)
        S = Xs - alpha * Ys
        trend = self._moving_average(S)
        S_c = S - S.mean(dim=1, keepdim=True)
        trend_c = trend - trend.mean(dim=1, keepdim=True)
        num = torch.mean(S_c * trend_c, dim=1, keepdim=True)
        den = torch.mean(trend_c * trend_c, dim=1, keepdim=True) + 1e-8
        lam = torch.clamp(num / den, 0.0, 1.0)
        gamma = torch.sigmoid(self.gamma_raw)
        return S - gamma * lam * trend_c


# ── MODEL: PlainKernel3 ────────────────────────────────────────────────────
# One temporal Conv1d with kernel=3. Output at t uses t-1, t, t+1.
# Initialized at green-only center tap.
# Params: 9
# RF: 3 samples
# ─────────────────────────────────────────────────────────────────────────────

class PlainKernel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 3
        self.dilation = 1
        self.proj = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3,
                              dilation=1, padding=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[0, 1, 1] = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        return self.proj(x_norm).squeeze(1)

    def reach_samples(self) -> int:
        return 3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STANDALONE TCNs
# Source: model_ZOO_primitive.py
#
# Three families:
#   SmallDilatedTCN   — simple residual dilated conv (one conv per block, non-causal)
#   CausalDilatedTCN  — same but causal (left-only padding)
#   TCN               — canonical Bai et al. 2018 (two convs per block, weight-norm)
# ═══════════════════════════════════════════════════════════════════════════════


# ── MODEL: SmallDilatedTCN ──────────────────────────────────────────────────
# Simple non-causal dilated TCN. One conv per block, symmetric padding.
# RF = 1 + (k-1) * sum(dilations)
# ─────────────────────────────────────────────────────────────────────────────

class SmallDilatedTCN(nn.Module):
    def __init__(self, hidden: int = 8, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.proj = nn.Conv1d(in_channels=3, out_channels=self.hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            padding = dilation * (self.kernel_size - 1) // 2
            self.blocks.append(nn.Conv1d(
                in_channels=self.hidden, out_channels=self.hidden,
                kernel_size=self.kernel_size, dilation=dilation,
                padding=padding, bias=True,
            ))
        self.head = nn.Conv1d(in_channels=self.hidden, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        h = self.proj(x_norm)
        for block in self.blocks:
            h = h + torch.relu(block(h))
        return self.head(h).squeeze(1)

    def reach_samples(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


# ── MODEL: CausalDilatedTCN ────────────────────────────────────────────────
# Same as SmallDilatedTCN but causal: left-only padding, no future leakage.
# RF = 1 + (k-1) * sum(dilations)
# ─────────────────────────────────────────────────────────────────────────────

class CausalDilatedTCN(nn.Module):
    def __init__(self, hidden: int = 8, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.proj = nn.Conv1d(in_channels=3, out_channels=self.hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            self.blocks.append(nn.Conv1d(
                in_channels=self.hidden, out_channels=self.hidden,
                kernel_size=self.kernel_size, dilation=dilation,
                padding=0, bias=True,
            ))
        self.head = nn.Conv1d(in_channels=self.hidden, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = mean_normalize_rgb(x)
        h = self.proj(x_norm)
        for block, dilation in zip(self.blocks, self.dilations):
            left = dilation * (self.kernel_size - 1)
            h_pad = F.pad(h, (left, 0))
            h = h + torch.relu(block(h_pad))
        return self.head(h).squeeze(1)

    def reach_samples(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


# ── MODEL: TCN (Canonical, Bai et al. 2018) ────────────────────────────────
# Two-conv-per-block residual TCN with weight-norm and dropout.
# Uses the shared TemporalBlock defined above.
# RF = 1 + 2*(k-1)*(2^n_blocks - 1)
#   n=3 → RF=29,  n=4 → RF=61
# Params: depends on hidden width (default h=16)
# ─────────────────────────────────────────────────────────────────────────────

class TCN(nn.Module):
    def __init__(self, n_blocks: int = 4, hidden: int = 16, kernel_size: int = 3,
                 dropout: float = 0.0, causal: bool = False, in_ch: int = 3):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        blocks = []
        for i in range(self.n_blocks):
            dilation = 2 ** i
            c_in = in_ch if i == 0 else hidden
            blocks.append(TemporalBlock(c_in, hidden, kernel_size, dilation, dropout, causal))
        self.network = nn.Sequential(*blocks)
        head = nn.Conv1d(hidden, 1, kernel_size=1)
        nn.init.normal_(head.weight, 0.0, 0.01)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = mean_normalize_rgb(x)
        h = self.network(xn)
        s = self.head(h)
        return s.squeeze(1)

    def receptive_field(self) -> int:
        total_dilation = sum(2 ** i for i in range(self.n_blocks))
        return 1 + 2 * (self.kernel_size - 1) * total_dilation

    def reach_samples(self) -> int:
        return self.receptive_field()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STANDALONE UNets
# Source: model_ZOO_unet_family.py + model_ZOO_multi_branch.py
#
# Three sizes sharing the same U-Net topology (3 encoder levels + bottleneck):
#   MiUNetStandalone    16/32/64/128   ~168k params
#   MMiUNetStandalone    8/16/32/64    ~42k params
#   MMMiUNetStandalone   4/8/16/32     ~10k params
# All have RF=96 and use the shared _unet_forward.
# ═══════════════════════════════════════════════════════════════════════════════


# ── MODEL: MiUNetStandalone ─────────────────────────────────────────────────
# Channels: 16/32/64/128. ~168,401 params. RF=96.
# Source: model_ZOO_multi_branch.py (MiUNetStandalone)
# ─────────────────────────────────────────────────────────────────────────────

class MiUNetStandalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = _unet_conv_block(3, 16);  self.enc2 = _unet_conv_block(16, 32)
        self.enc3 = _unet_conv_block(32, 64); self.pool = nn.MaxPool1d(2)
        self.bottleneck = _unet_conv_block(64, 128)
        self.up3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3 = _unet_conv_block(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = _unet_conv_block(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = _unet_conv_block(32, 16)
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x):
        x = mean_normalize_rgb(x)
        d1 = _unet_forward(self.enc1, self.enc2, self.enc3, self.pool, self.bottleneck,
                           self.up3, self.dec3, self.up2, self.dec2, self.up1, self.dec1, x)
        return self.final_conv(d1).squeeze(1)

    def reach_samples(self):
        return 96


# ── MODEL: MMiUNetStandalone ────────────────────────────────────────────────
# Channels: 8/16/32/64. ~42,345 params. RF=96.
# Source: model_ZOO_multi_branch.py (MMiUNetStandalone)
# ─────────────────────────────────────────────────────────────────────────────

class MMiUNetStandalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = _unet_conv_block(3, 8);   self.enc2 = _unet_conv_block(8, 16)
        self.enc3 = _unet_conv_block(16, 32); self.pool = nn.MaxPool1d(2)
        self.bottleneck = _unet_conv_block(32, 64)
        self.up3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec3 = _unet_conv_block(64, 32)
        self.up2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec2 = _unet_conv_block(32, 16)
        self.up1 = nn.ConvTranspose1d(16, 8, 2, stride=2)
        self.dec1 = _unet_conv_block(16, 8)
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)

    def forward(self, x):
        x = mean_normalize_rgb(x)
        d1 = _unet_forward(self.enc1, self.enc2, self.enc3, self.pool, self.bottleneck,
                           self.up3, self.dec3, self.up2, self.dec2, self.up1, self.dec1, x)
        return self.final_conv(d1).squeeze(1)

    def reach_samples(self):
        return 96


# ── MODEL: MMMiUNetStandalone ───────────────────────────────────────────────
# Channels: 4/8/16/32. ~10,709 params. RF=96.
# Source: model_ZOO_unet_family.py (MMMiUNet)
# ─────────────────────────────────────────────────────────────────────────────

class MMMiUNetStandalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = _unet_conv_block(3, 4);   self.enc2 = _unet_conv_block(4, 8)
        self.enc3 = _unet_conv_block(8, 16);  self.pool = nn.MaxPool1d(2)
        self.bottleneck = _unet_conv_block(16, 32)
        self.up3 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec3 = _unet_conv_block(32, 16)
        self.up2 = nn.ConvTranspose1d(16, 8, 2, stride=2)
        self.dec2 = _unet_conv_block(16, 8)
        self.up1 = nn.ConvTranspose1d(8, 4, 2, stride=2)
        self.dec1 = _unet_conv_block(8, 4)
        self.final_conv = nn.Conv1d(4, 1, kernel_size=1)

    def forward(self, x):
        x = mean_normalize_rgb(x)
        d1 = _unet_forward(self.enc1, self.enc2, self.enc3, self.pool, self.bottleneck,
                           self.up3, self.dec3, self.up2, self.dec2, self.up1, self.dec1, x)
        return self.final_conv(d1).squeeze(1)

    def reach_samples(self):
        return 96


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STANDALONE FREQUENCY-DOMAIN MODELS
# Source: model_primitive_0.py
#
# Models that operate in the frequency domain (FFT-based processing).
# ═══════════════════════════════════════════════════════════════════════════════


# ── BUILDING BLOCK: Frequencydomain_FFN ─────────────────────────────────────
# Frequency-domain feedforward block.
# fc1 projects channels up, FFT over time, spectral mixing, IFFT, fc2 back.
# Used as a building block inside the FFN model.
# ─────────────────────────────────────────────────────────────────────────────

class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2):
        super().__init__()
        self.scale = 0.02
        self.hidden_dim = dim * mlp_ratio
        self.r = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.hidden_dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.hidden_dim))
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x_f = torch.fft.fft(x, dim=1, norm="ortho")
        x_real = F.relu(
            torch.einsum("btc,cc->btc", x_f.real, self.r)
            - torch.einsum("btc,cc->btc", x_f.imag, self.i)
            + self.rb
        )
        x_imag = F.relu(
            torch.einsum("btc,cc->btc", x_f.imag, self.r)
            + torch.einsum("btc,cc->btc", x_f.real, self.i)
            + self.ib
        )
        x_f = torch.stack([x_real, x_imag], dim=-1).float()
        x_f = torch.view_as_complex(x_f)
        x = torch.fft.ifft(x_f, dim=1, norm="ortho").real
        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


# ── MODEL: FFN ──────────────────────────────────────────────────────────────
# Single-ROI RGB model using frequency-domain processing.
# InstanceNorm input → channel projection → freq block → time-domain refine → output
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, hidden: int = 8, mlp_ratio: int = 2, use_input_norm: bool = True):
        super().__init__()
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.input_norm = nn.InstanceNorm1d(num_features=3, affine=True)
        self.input_proj = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=1, bias=True), nn.GELU(),
        )
        self.freq_block = Frequencydomain_FFN(dim=hidden, mlp_ratio=mlp_ratio)
        self.refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=True), nn.GELU(),
        )
        self.output_proj = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()
        if self.use_input_norm:
            x = self.input_norm(x)
        x = self.input_proj(x)
        x_t = x.transpose(1, 2).contiguous()
        x_t = x_t + self.freq_block(x_t)
        x = x_t.transpose(1, 2).contiguous()
        x = x + self.refine(x)
        y = self.output_proj(x)
        return y.squeeze(1)


# ── BUILDING BLOCK: ComplexSpectralMixer ────────────────────────────────────
# Learnable complex spectral mixer (LCSM). FNO-style complex channel mixing
# in frequency domain with proper phase handling.
# Used as a building block inside BVPNet_V2.
# ─────────────────────────────────────────────────────────────────────────────

class ComplexSpectralMixer(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2, scale: float = 0.02):
        super().__init__()
        H = dim * mlp_ratio
        self.W_r = nn.Parameter(scale * torch.randn(H, H))
        self.W_i = nn.Parameter(scale * torch.randn(H, H))
        self.proj_up = nn.Sequential(
            nn.Conv1d(dim, H, kernel_size=1, bias=False),
            nn.ELU(),
        )
        self.proj_down = nn.Conv1d(H, dim, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm1d(dim, affine=True)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        residual = x
        h = self.proj_up(x)
        xf = torch.fft.rfft(h, dim=-1)
        out_re = (torch.einsum("bcf,hc->bhf", xf.real, self.W_r)
                  - torch.einsum("bcf,hc->bhf", xf.imag, self.W_i))
        out_im = (torch.einsum("bcf,hc->bhf", xf.real, self.W_i)
                  + torch.einsum("bcf,hc->bhf", xf.imag, self.W_r))
        xf_out = torch.complex(out_re, out_im)
        h = torch.fft.irfft(xf_out, n=T, dim=-1)
        h = self.act(h)
        h = self.norm(self.proj_down(h))
        return residual + h


# ── BUILDING BLOCK: TemporalSE ─────────────────────────────────────────────
# Temporal squeeze-and-excitation channel gating.
# Used as a building block inside BVPNet_V2.
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(x).unsqueeze(-1)
        return x * scale


# ── MODEL: BVPNet_V2 ───────────────────────────────────────────────────────
# Compact rPPG network with ComplexSpectralMixer + TemporalSE.
# Input is temporal-differenced before processing.
# Architecture: InstanceNorm → proj → local conv → dilated conv
#             → ComplexSpectralMixer → TemporalSE → output
# ─────────────────────────────────────────────────────────────────────────────

class BVPNet_V2(nn.Module):
    def __init__(self, hidden: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(num_features=3, affine=True)
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden, kernel_size=1, bias=True),
            nn.ELU(),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden, out_channels=hidden,
                      kernel_size=7, padding=3, bias=True),
            nn.ELU(),
        )
        self.dil_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden, out_channels=hidden,
                      kernel_size=7, dilation=2, padding=6, bias=True),
            nn.ELU(),
        )
        self.spectral = ComplexSpectralMixer(dim=hidden, mlp_ratio=mlp_ratio)
        self.se = TemporalSE(channels=hidden, reduction=2)
        self.out_proj = nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.diff(x, dim=1)
        x = F.pad(x, (0, 0, 1, 0))
        x = x.permute(0, 2, 1).contiguous()
        x = self.input_norm(x)
        x = self.proj(x)
        x = x + self.local_conv(x)
        x = x + self.dil_conv(x)
        x = self.spectral(x)
        x = self.se(x)
        x = self.out_proj(x)
        return x.squeeze(1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — THREE-BRANCH FUSION SYSTEM
#
# Components:
#   Branch 1 backbones — RGB projectors (1ch or 2ch output)
#   Branch 2 backbones — TCN feature extractors (8ch output)
#   Branch 3 backbones — UNet feature extractors (8ch or 16ch output)
#   Standalone Branch 2 — TCN with head (for standalone evaluation)
#   Fusion strategies  — 5 types, all accept (feat, f1, f2, f3)
#   ThreeBranchModel   — container that wires branches + fusion
#
# Source: model_ZOO_multi_branch.py (original fusion system)
#       + model_ZOO_FNN.py (6-weight and derivative branch additions)
#
# BACKWARD COMPATIBILITY:
#   ThreeBranchModel("se_attention", "mi") produces the EXACT same model
#   as the original model_ZOO_multi_branch.py. Same parameter creation
#   order, same forward pass, same fusion signatures.
# ═══════════════════════════════════════════════════════════════════════════════


# ── BRANCH 1: 3-weight projector (ORIGINAL) ────────────────────────────────
# Single pointwise RGB projection. Output: [B, 1, T].
# Source: model_ZOO_multi_branch.py (Branch1Backbone)
# ─────────────────────────────────────────────────────────────────────────────

class Branch1Backbone(nn.Module):
    """Branch 1 backbone for fusion — 3 weights, 1ch output. RF=1."""
    out_ch = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor([[[0.0], [-1.0], [0.0]]], dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(mean_normalize_rgb(x))


# ── BRANCH 1: 6-weight + learnable alpha (NEW) ─────────────────────────────
# Two projections combined inside B1: output = Xs - α * Ys
# α is a learnable parameter (initialized at 1.0).
# Output: [B, 1, T] — same as 3w B1.
# ─────────────────────────────────────────────────────────────────────────────

class Branch1_6w_alpha_Backbone(nn.Module):
    """Branch 1 backbone — 6 weights + 1 learnable alpha, 1ch output. RF=1."""
    out_ch = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([
                [[ 0.0], [-1.0], [ 0.0]],
                [[-1.0], [ 0.0], [ 1.0]],
            ], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(mean_normalize_rgb(x))
        Xs = h[:, 0:1, :]
        Ys = h[:, 1:2, :]
        return Xs - self.alpha * Ys





# ── BRANCH 1: 6-weight dual projector (NEW) ────────────────────────────────
# Two pointwise RGB projections. Output: [B, 2, T].
# Learns pulse-aligned + noise-reference projections.
# Source: model_ZOO_FNN.py (Branch1_6wBackbone)
# ─────────────────────────────────────────────────────────────────────────────

class Branch1_6wBackbone(nn.Module):
    """Branch 1 backbone — 6 weights, 2ch output. RF=1."""
    out_ch = 2

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([
                [[ 0.0], [-1.0], [ 0.0]],
                [[-1.0], [ 0.0], [ 1.0]],
            ], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(mean_normalize_rgb(x))


# ── BRANCH 2: Raw TCN (ORIGINAL) ───────────────────────────────────────────
# Canonical TCN on mean-normalized RGB. Output: [B, 8, T].
# Source: model_ZOO_multi_branch.py (Branch2Backbone)
# ─────────────────────────────────────────────────────────────────────────────

class Branch2Backbone(nn.Module):
    """Branch 2 backbone — TCN on raw RGB, 8ch output. RF=29."""
    out_ch = 8

    def __init__(self, n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False, in_ch=3):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        blocks = []
        for i in range(self.n_blocks):
            c_in = in_ch if i == 0 else hidden
            blocks.append(TemporalBlock(c_in, hidden, kernel_size, 2**i, dropout, causal))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(mean_normalize_rgb(x))


# ── BRANCH 2: Derivative TCN (NEW) ─────────────────────────────────────────
# TCN on first-difference of RGB. Amplifies pulse edges, removes drift.
# Output: [B, 8, T].
# Source: model_ZOO_FNN.py (Branch2DerivBackbone)
# ─────────────────────────────────────────────────────────────────────────────

class Branch2DerivBackbone(nn.Module):
    """Branch 2 backbone — TCN on temporal derivative of RGB, 8ch output. RF=29."""
    out_ch = 8

    def __init__(self, n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False, in_ch=3):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        blocks = []
        for i in range(self.n_blocks):
            c_in = in_ch if i == 0 else hidden
            blocks.append(TemporalBlock(c_in, hidden, kernel_size, 2**i, dropout, causal))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        h = mean_normalize_rgb(x)
        diff = h[:, :, 1:] - h[:, :, :-1]
        diff = F.pad(diff, (1, 0), mode='constant', value=0.0)
        return self.network(diff)


# ── BRANCH 2: Standalone TCN with head ──────────────────────────────────────
# Same TCN as Branch2Backbone but with a 1x1 head for standalone evaluation.
# Output: [B, T].
# Source: model_ZOO_multi_branch.py (TCNStandalone)
# ─────────────────────────────────────────────────────────────────────────────

class TCNStandalone(nn.Module):
    """Standalone Branch 2 — TCN with output head. Returns [B, T]."""
    def __init__(self, n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False, in_ch=3):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        blocks = []
        for i in range(self.n_blocks):
            c_in = in_ch if i == 0 else hidden
            blocks.append(TemporalBlock(c_in, hidden, kernel_size, 2**i, dropout, causal))
        self.network = nn.Sequential(*blocks)
        head = nn.Conv1d(hidden, 1, kernel_size=1)
        nn.init.normal_(head.weight, 0.0, 0.01)
        self.head = head

    def forward(self, x):
        return self.head(self.network(mean_normalize_rgb(x))).squeeze(1)

    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * sum(2**i for i in range(self.n_blocks))

    def reach_samples(self):
        return self.receptive_field()


# ── BRANCH 3: MiUNet backbone ──────────────────────────────────────────────
# MiUNet without final conv, for fusion. Output: [B, 16, T].
# Source: model_ZOO_multi_branch.py (MiUNetBackbone)
# ─────────────────────────────────────────────────────────────────────────────

class MiUNetBackbone(nn.Module):
    """MiUNet backbone for fusion — 16/32/64/128, 16ch output."""
    out_ch = 16

    def __init__(self):
        super().__init__()
        self.enc1 = _unet_conv_block(3, 16);  self.enc2 = _unet_conv_block(16, 32)
        self.enc3 = _unet_conv_block(32, 64); self.pool = nn.MaxPool1d(2)
        self.bottleneck = _unet_conv_block(64, 128)
        self.up3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3 = _unet_conv_block(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = _unet_conv_block(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = _unet_conv_block(32, 16)

    def forward(self, x):
        x = mean_normalize_rgb(x)
        return _unet_forward(self.enc1, self.enc2, self.enc3, self.pool,
                             self.bottleneck, self.up3, self.dec3, self.up2,
                             self.dec2, self.up1, self.dec1, x)


# ── BRANCH 3: MMiUNet backbone ─────────────────────────────────────────────
# MMiUNet without final conv, for fusion. Output: [B, 8, T].
# Source: model_ZOO_multi_branch.py (MMiUNetBackbone)
# ─────────────────────────────────────────────────────────────────────────────

class MMiUNetBackbone(nn.Module):
    """MMiUNet backbone for fusion — 8/16/32/64, 8ch output."""
    out_ch = 8

    def __init__(self):
        super().__init__()
        self.enc1 = _unet_conv_block(3, 8);   self.enc2 = _unet_conv_block(8, 16)
        self.enc3 = _unet_conv_block(16, 32); self.pool = nn.MaxPool1d(2)
        self.bottleneck = _unet_conv_block(32, 64)
        self.up3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec3 = _unet_conv_block(64, 32)
        self.up2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec2 = _unet_conv_block(32, 16)
        self.up1 = nn.ConvTranspose1d(16, 8, 2, stride=2)
        self.dec1 = _unet_conv_block(16, 8)

    def forward(self, x):
        x = mean_normalize_rgb(x)
        return _unet_forward(self.enc1, self.enc2, self.enc3, self.pool,
                             self.bottleneck, self.up3, self.dec3, self.up2,
                             self.dec2, self.up1, self.dec1, x)


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION STRATEGIES
# Source: model_ZOO_multi_branch.py (ALL 5 strategies, ORIGINAL signatures)
#
# All fusion classes accept (total_ch, branch3_ch) in __init__
# and (feat, f1, f2, f3) in forward — even when some args are unused.
# This preserves backward compatibility with existing checkpoints.
# ═══════════════════════════════════════════════════════════════════════════════

BRANCH1_CH = 1    # default B1 output channels (3-weight projector)
BRANCH2_CH = 8    # B2 output channels (TCN hidden)


class FusionStaticLinear(nn.Module):
    """Strategy 1 — 1x1 Conv(C->1). No per-window adaptation."""
    def __init__(self, total_ch, branch3_ch):
        super().__init__()
        self.conv = nn.Conv1d(total_ch, 1, 1, bias=True)
    def forward(self, feat, f1, f2, f3):
        return self.conv(feat)


class FusionTwoLayerMLP(nn.Module):
    """Strategy 2 — 1x1 Conv(C->8) + ReLU + 1x1 Conv(8->1)."""
    def __init__(self, total_ch, branch3_ch, mid=8):
        super().__init__()
        self.fc1 = nn.Conv1d(total_ch, mid, 1, bias=True)
        self.fc2 = nn.Conv1d(mid, 1, 1, bias=True)
    def forward(self, feat, f1, f2, f3):
        return self.fc2(F.relu(self.fc1(feat)))


class FusionSEAttention(nn.Module):
    """Strategy 3 — SE bottleneck channel gating."""
    def __init__(self, total_ch, branch3_ch, r=8):
        super().__init__()
        r = max(r, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(total_ch, r); self.fc2 = nn.Linear(r, total_ch)
        self.out = nn.Conv1d(total_ch, 1, 1, bias=True)
    def forward(self, feat, f1, f2, f3):
        s = self.gap(feat).squeeze(-1)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return self.out(feat * g.unsqueeze(-1))


class FusionLightweightAttention(nn.Module):
    """Strategy 4 — Single FC channel gating (no bottleneck)."""
    def __init__(self, total_ch, branch3_ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(total_ch, total_ch)
        self.out = nn.Conv1d(total_ch, 1, 1, bias=True)
    def forward(self, feat, f1, f2, f3):
        s = self.gap(feat).squeeze(-1)
        g = torch.sigmoid(self.fc(s))
        return self.out(feat * g.unsqueeze(-1))


class FusionBranchAttention(nn.Module):
    """Strategy 5 — 3-scalar softmax branch gating."""
    def __init__(self, total_ch, branch3_ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(total_ch, 3)
        self.head2 = nn.Conv1d(BRANCH2_CH, 1, 1, bias=True)
        self.head3 = nn.Conv1d(branch3_ch, 1, 1, bias=True)
    def forward(self, feat, f1, f2, f3):
        alpha = F.softmax(self.fc(self.gap(feat).squeeze(-1)), dim=-1)
        o1 = f1; o2 = self.head2(f2); o3 = self.head3(f3)
        a = alpha.unsqueeze(-1).unsqueeze(-1)
        return a[:,0]*o1 + a[:,1]*o2 + a[:,2]*o3


# ═══════════════════════════════════════════════════════════════════════════════
# THREE-BRANCH MODEL
#
# Source: model_ZOO_multi_branch.py (ORIGINAL), extended with optional
#         branch1_cls and branch2_cls for 6-weight and derivative variants.
#
# BACKWARD COMPATIBILITY:
#   ThreeBranchModel("se_attention", "mi")
#   → branch1_cls=None → Branch1Backbone (3w)
#   → branch2_cls=None → Branch2Backbone (raw TCN)
#   → Same parameter creation order as original
#   → Same total_ch = 1 + 8 + 16 = 25
#   → Same fusion constructed with (25, 16)
#   → Byte-identical model to model_ZOO_multi_branch.py
#
# NEW USAGE:
#   ThreeBranchModel("se_attention", "mi",
#                    branch1_cls=Branch1_6wBackbone,
#                    branch2_cls=Branch2DerivBackbone)
#   → total_ch = 2 + 8 + 16 = 26
# ═══════════════════════════════════════════════════════════════════════════════

_FUSION = {
    "static_linear":         FusionStaticLinear,
    "mlp":                   FusionTwoLayerMLP,
    "se_attention":          FusionSEAttention,
    "lightweight_attention": FusionLightweightAttention,
    "branch_attention":      FusionBranchAttention,
}
_BACKBONE = {"mi": MiUNetBackbone, "mmi": MMiUNetBackbone}


class ThreeBranchModel(nn.Module):
    """
    B1 [B, b1ch, T] + B2 [B, 8, T] + B3 [B, b3ch, T]
    → concat → fusion → [B, 1, T] → squeeze → [B, T]
    """
    def __init__(self, fusion_type, backbone_variant="mi",
                 branch1_cls=None, branch2_cls=None):
        super().__init__()
        # Branch 1 — default: 3-weight projector (original behavior)
        if branch1_cls is not None:
            self.branch1 = branch1_cls()
        else:
            self.branch1 = Branch1Backbone()

        # Branch 2 — default: raw RGB TCN (original behavior)
        if branch2_cls is not None:
            self.branch2 = branch2_cls(
                n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False)
        else:
            self.branch2 = Branch2Backbone(
                n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False)

        # Branch 3 — determined by backbone_variant
        self.branch3 = _BACKBONE[backbone_variant]()

        b1ch = getattr(self.branch1, 'out_ch', BRANCH1_CH)
        b3ch = self.branch3.out_ch
        total = b1ch + BRANCH2_CH + b3ch
        self.total_ch = total

        self.fusion = _FUSION[fusion_type](total, b3ch)
        self.fusion_type = fusion_type
        self.backbone_variant = backbone_variant

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        return self.fusion(torch.cat([f1, f2, f3], dim=1), f1, f2, f3).squeeze(1)

    def reach_samples(self):
        return 96


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
#
# Every model accessible via build_model(name).
# Organized by family. Old names preserved for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════════════

_HIDDEN_TCN = 16
_KSIZE = 3
_DROP = 0.0

MODEL_REGISTRY = {

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 1 — STANDALONE PROJECTORS
    # ══════════════════════════════════════════════════════════════════════════
    "two_projection_CHROM_ADAPTIVE":  lambda: TwoProjectionRGBProjector(),      # 6 params, Xs-αYs
    "plain_k1":                       lambda: SimpleRGBProjector(),              # 3 params, [0,-1,0]
    "adaptive_trend":                 lambda: AdaptiveTrendProjector(trend_kernel=31),
    "two_branch_adaptive":            lambda: TwoBranchAdaptive(K=31),           # 6 params, pulse+nuisance
    "adaptive_detrend":               lambda: AdaptiveDetrendProjector(trend_kernel=61, gamma_init=0.5),
    "plain_k3":                       lambda: PlainKernel3(),                    # 9 params, kernel=3

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 2 — SIMPLE DILATED TCN (non-causal)
    # ══════════════════════════════════════════════════════════════════════════
    "simple_tcn_rf31_h4":             lambda: SmallDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8)),
    "simple_tcn_rf61_h4":             lambda: SmallDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8,15)),
    "simple_tcn_rf121_h4":            lambda: SmallDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8,15,30)),
    "simple_tcn_rf31_h8":             lambda: SmallDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8)),
    "simple_tcn_rf61_h8":             lambda: SmallDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8,15)),
    "simple_tcn_rf121_h8":            lambda: SmallDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8,15,30)),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 3 — SIMPLE DILATED TCN (causal)
    # ══════════════════════════════════════════════════════════════════════════
    "simple_tcn_rf31_h4_causal":      lambda: CausalDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8)),
    "simple_tcn_rf61_h4_causal":      lambda: CausalDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8,15)),
    "simple_tcn_rf121_h4_causal":     lambda: CausalDilatedTCN(hidden=4, kernel_size=3, dilations=(1,2,4,8,15,30)),
    "simple_tcn_rf31_h8_causal":      lambda: CausalDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8)),
    "simple_tcn_rf61_h8_causal":      lambda: CausalDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8,15)),
    "simple_tcn_rf121_h8_causal":     lambda: CausalDilatedTCN(hidden=8, kernel_size=3, dilations=(1,2,4,8,15,30)),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 4 — CANONICAL TCN (Bai et al. 2018)
    # ══════════════════════════════════════════════════════════════════════════
    "canonical_tcn_rf31":             lambda: TCN(n_blocks=3, hidden=_HIDDEN_TCN, kernel_size=_KSIZE, dropout=_DROP, causal=False),
    "canonical_tcn_rf61":             lambda: TCN(n_blocks=4, hidden=_HIDDEN_TCN, kernel_size=_KSIZE, dropout=_DROP, causal=False),
    "canonical_tcn_rf31_causal":      lambda: TCN(n_blocks=3, hidden=_HIDDEN_TCN, kernel_size=_KSIZE, dropout=_DROP, causal=True),
    "canonical_tcn_rf61_causal":      lambda: TCN(n_blocks=4, hidden=_HIDDEN_TCN, kernel_size=_KSIZE, dropout=_DROP, causal=True),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 5 — STANDALONE UNets
    # ══════════════════════════════════════════════════════════════════════════
    "MiUNet":                         lambda: MiUNetStandalone(),                # 16/32/64/128, ~168k
    "MMiUNet":                        lambda: MMiUNetStandalone(),               # 8/16/32/64, ~42k
    "MMMiUNet":                       lambda: MMMiUNetStandalone(),              # 4/8/16/32, ~10k

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 6 — STANDALONE FREQUENCY-DOMAIN MODELS
    # ══════════════════════════════════════════════════════════════════════════
    "FFN":                            lambda: FFN(hidden=8, mlp_ratio=2),
    "BVPNet_V2":                      lambda: BVPNet_V2(hidden=4, mlp_ratio=2),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 7 — STANDALONE BRANCH COMPONENTS (for isolated evaluation)
    # ══════════════════════════════════════════════════════════════════════════
    "branch1_projector":              lambda: SimpleRGBProjector(),              # same as plain_k1
    "branch2_tcn_rf31_h8":            lambda: TCNStandalone(n_blocks=3, hidden=8, kernel_size=3, dropout=0.0, causal=False),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 8 — THREE-BRANCH FUSION: PRIMITIVE (B1=3w, B2=raw TCN)
    #
    # Original model names from model_ZOO_multi_branch.py.
    # ThreeBranchModel("fusion_type", "backbone_variant")
    # B1=Branch1Backbone(3w, 1ch), B2=Branch2Backbone(raw, 8ch)
    # ══════════════════════════════════════════════════════════════════════════

    # ── MiUNet backbone (B3=16ch, total=1+8+16=25ch) ────────────────────────
    "three_branch_mi_static_linear":         lambda: ThreeBranchModel("static_linear", "mi"),
    "three_branch_mi_mlp":                   lambda: ThreeBranchModel("mlp", "mi"),
    "three_branch_mi_se_attention":          lambda: ThreeBranchModel("se_attention", "mi"),
    "three_branch_mi_lightweight_attention": lambda: ThreeBranchModel("lightweight_attention", "mi"),
    "three_branch_mi_branch_attention":      lambda: ThreeBranchModel("branch_attention", "mi"),

    # ── MMiUNet backbone (B3=8ch, total=1+8+8=17ch) ─────────────────────────
    "three_branch_mmi_static_linear":         lambda: ThreeBranchModel("static_linear", "mmi"),
    "three_branch_mmi_mlp":                   lambda: ThreeBranchModel("mlp", "mmi"),
    "three_branch_mmi_se_attention":          lambda: ThreeBranchModel("se_attention", "mmi"),
    "three_branch_mmi_lightweight_attention": lambda: ThreeBranchModel("lightweight_attention", "mmi"),
    "three_branch_mmi_branch_attention":      lambda: ThreeBranchModel("branch_attention", "mmi"),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 9 — THREE-BRANCH FUSION: 6-WEIGHT B1 + DERIVATIVE B2
    #
    # B1=Branch1_6wBackbone(6w, 2ch), B2=Branch2DerivBackbone(deriv, 8ch)
    # ══════════════════════════════════════════════════════════════════════════

    # ── MiUNet backbone (B3=16ch, total=2+8+16=26ch) ────────────────────────
    "b1_6w_b2_deriv_b3_mi__SL":  lambda: ThreeBranchModel("static_linear", "mi",
                                          branch1_cls=Branch1_6wBackbone, branch2_cls=Branch2DerivBackbone),
    "b1_6w_b2_deriv_b3_mi__SE":  lambda: ThreeBranchModel("se_attention", "mi",
                                          branch1_cls=Branch1_6wBackbone, branch2_cls=Branch2DerivBackbone),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 10 — THREE-BRANCH FUSION: PRIMITIVE B1/B2, EXPLICIT NAMES
    #
    # Same models as Family 8 (MiUNet only), but with systematic naming
    # that matches the margin sweep experiment convention.
    # ══════════════════════════════════════════════════════════════════════════
    "b1_3w_b2_raw_b3_mi__SL":   lambda: ThreeBranchModel("static_linear", "mi"),
    "b1_3w_b2_raw_b3_mi__SE":   lambda: ThreeBranchModel("se_attention", "mi"),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 11 — THREE-BRANCH FUSION: 6-WEIGHT B1 + RAW B2
    #
    # B1=Branch1_6wBackbone(6w, 2ch), B2=Branch2Backbone(raw, 8ch)
    # Only B1 changed vs Family 8 — clean comparison.
    # ══════════════════════════════════════════════════════════════════════════
    "b1_6w_b2_raw_b3_mi__SL":   lambda: ThreeBranchModel("static_linear", "mi",
                                         branch1_cls=Branch1_6wBackbone),
    "b1_6w_b2_raw_b3_mi__SE":   lambda: ThreeBranchModel("se_attention", "mi",
                                         branch1_cls=Branch1_6wBackbone),

    # ══════════════════════════════════════════════════════════════════════════
    # FAMILY 12 — THREE-BRANCH FUSION: 6-WEIGHT+ALPHA B1 + RAW B2
    #
    # B1=Branch1_6w_alpha_Backbone(7 params, 1ch), B2=Branch2Backbone(raw, 8ch)
    # Output = Xs - alpha * Ys (learnable CHROM-like combination inside B1)
    # ══════════════════════════════════════════════════════════════════════════
    "b1_6wa_b2_raw_b3_mi__SL":  lambda: ThreeBranchModel("static_linear", "mi",
                                         branch1_cls=Branch1_6w_alpha_Backbone),
    "b1_6wa_b2_raw_b3_mi__SE":  lambda: ThreeBranchModel("se_attention", "mi",
                                         branch1_cls=Branch1_6w_alpha_Backbone),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ORDERS — organized by family
#
# SWEEP_ORDER is the default for longNight_TRAIN.py.
# Select a family by assigning: SWEEP_ORDER = SWEEP_ORDER_<family>
# ═══════════════════════════════════════════════════════════════════════════════

# ── Family 1: standalone projectors ─────────────────────────────────────────
SWEEP_ORDER_PROJECTORS = [
    "two_projection_CHROM_ADAPTIVE",
    "plain_k1",
    "adaptive_trend",
    "two_branch_adaptive",
    "adaptive_detrend",
    "plain_k3",
]

# ── Family 2+3: simple dilated TCNs ─────────────────────────────────────────
SWEEP_ORDER_SIMPLE_TCN = [
    "simple_tcn_rf31_h4", "simple_tcn_rf61_h4", "simple_tcn_rf121_h4",
    "simple_tcn_rf31_h8", "simple_tcn_rf61_h8", "simple_tcn_rf121_h8",
    "simple_tcn_rf31_h4_causal", "simple_tcn_rf61_h4_causal", "simple_tcn_rf121_h4_causal",
    "simple_tcn_rf31_h8_causal", "simple_tcn_rf61_h8_causal", "simple_tcn_rf121_h8_causal",
]

# ── Family 4: canonical TCNs ────────────────────────────────────────────────
SWEEP_ORDER_CANONICAL_TCN = [
    "canonical_tcn_rf31", "canonical_tcn_rf61",
    "canonical_tcn_rf31_causal", "canonical_tcn_rf61_causal",
]

# ── Family 5: standalone UNets ──────────────────────────────────────────────
SWEEP_ORDER_UNET = [
    "MiUNet", "MMiUNet", "MMMiUNet",
]

# ── Family 6: frequency-domain models ───────────────────────────────────────
SWEEP_ORDER_FREQ = [
    "FFN", "BVPNet_V2",
]

# ── Family 8: three-branch fusion (primitive, MiUNet) ───────────────────────
SWEEP_ORDER_FUSION_MI = [
    "three_branch_mi_static_linear",
    "three_branch_mi_mlp",
    "three_branch_mi_se_attention",
    "three_branch_mi_lightweight_attention",
    "three_branch_mi_branch_attention",
]

# ── Family 8: three-branch fusion (primitive, MMiUNet) ──────────────────────
SWEEP_ORDER_FUSION_MMI = [
    "three_branch_mmi_static_linear",
    "three_branch_mmi_mlp",
    "three_branch_mmi_se_attention",
    "three_branch_mmi_lightweight_attention",
    "three_branch_mmi_branch_attention",
]

# ── Family 9: three-branch fusion (6w + derivative) ────────────────────────
SWEEP_ORDER_FUSION_6W = [
    "b1_6w_b2_deriv_b3_mi__SL",
    "b1_6w_b2_deriv_b3_mi__SE",
]


# ── Family 10: three-branch fusion (primitive, explicit names) ──────────────
SWEEP_ORDER_FUSION_PRIMITIVE = [
    "b1_3w_b2_raw_b3_mi__SL",
    "b1_3w_b2_raw_b3_mi__SE",
]

# ── Family 11: three-branch fusion (6w B1 + raw B2) ────────────────────────
SWEEP_ORDER_FUSION_6W_RAW = [
    "b1_6w_b2_raw_b3_mi__SL",
    "b1_6w_b2_raw_b3_mi__SE",
]

# ── Family 12: three-branch fusion (6w+alpha B1 + raw B2) ──────────────────
SWEEP_ORDER = [
    "b1_3w_b2_raw_b3_mi__SE"
]



# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT SWEEP ORDER — change this to select which family to train
# ═════════════════════════════════════════════════════════════════════════════

#SWEEP_ORDER = SWEEP_ORDER_FUSION_MI + SWEEP_ORDER_FUSION_6W + SWEEP_ORDER_FUSION_6W_RAW

SWEEP_ORDER = [
    "three_branch_mi_se_attention",       #Current best model
]




# ═══════════════════════════════════════════════════════════════════════════════
# build_model — the single entry point
# ══
def build_model(name: str) -> nn.Module:
    """Construct a fresh model instance by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, T = 2, 240
    dummy = torch.randn(B, T, 3)

    print(f"\n{'model':<45} {'params':>10} {'reach':>8}   out")
    print("-" * 80)

    failed = []
    for name in sorted(MODEL_REGISTRY.keys()):
        try:
            m = build_model(name)
            m.eval()
            with torch.no_grad():
                out = m(dummy)
            assert out.shape == (B, T), f"{name}: got {out.shape}"
            n_params = sum(p.numel() for p in m.parameters())
            r = m.reach_samples() if hasattr(m, "reach_samples") else "-"
            print(f"{name:<45} {n_params:>10} {str(r):>8}   {tuple(out.shape)}")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"{name:<45} FAILED: {e}")

    # Odd-length robustness (UNet padding)
    for T_odd in [117, 185, 233]:
        for name in sorted(MODEL_REGISTRY.keys()):
            try:
                m = build_model(name)
                m.eval()
                with torch.no_grad():
                    o = m(torch.randn(1, T_odd, 3))
                assert o.shape == (1, T_odd), f"{name} T={T_odd}: {o.shape}"
            except Exception as e:
                if (name, str(e)) not in failed:
                    failed.append((name, f"T={T_odd}: {e}"))

    if failed:
        print(f"\n{len(failed)} FAILURES:")
        for n, e in failed:
            print(f"  {n}: {e}")
    else:
        print(f"\nAll {len(MODEL_REGISTRY)} models pass. Self-test complete.")
