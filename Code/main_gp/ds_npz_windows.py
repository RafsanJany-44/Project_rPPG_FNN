# ds_npz_windows.py
# Our NPZ window dataset (PURE / UBFC / UBFC-Phys)
# We sample windows on-the-fly (no overlapped windows on disk).

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class WindowSpec:
    win_s: float = 8.0
    stride_s: float = 1.0
    max_windows_per_seq: int = 80  # our cap so one long video doesn't dominate
    fs_ref: Optional[float] = None  # if set, we force global fs


def _estimate_fs_from_t(t: np.ndarray) -> float:
    if t is None or len(t) < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    return 1.0 / float(np.median(dt))


def _load_meta(z: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "meta" not in z:
        return {}
    meta_raw = z["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8", errors="ignore")
    if isinstance(meta_raw, str):
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}
    if isinstance(meta_raw, dict):
        return meta_raw
    return {}


def _get_fs_from_meta_or_t(meta: Dict[str, Any], t: np.ndarray) -> float:
    # Prefer fps stored in meta (UBFC / UBFC-Phys exporters should include it)
    fps = meta.get("fps", None)
    try:
        fps = float(fps) if fps is not None else float("nan")
    except Exception:
        fps = float("nan")

    if np.isfinite(fps) and fps > 1.0:
        return float(fps)

    # PURE fallback: estimate from timestamps
    return float(_estimate_fs_from_t(t))


def _pad_or_trim_1d(x: np.ndarray, L: int) -> np.ndarray:
    T = int(x.shape[0])
    if T == L:
        return x
    if T > L:
        return x[:L]
    out = np.zeros((L,), dtype=x.dtype)
    out[:T] = x
    return out


def _pad_or_trim_2d(x: np.ndarray, L: int) -> np.ndarray:
    T = int(x.shape[0])
    C = int(x.shape[1])
    if T == L:
        return x
    if T > L:
        return x[:L, :]
    out = np.zeros((L, C), dtype=x.dtype)
    out[:T, :] = x
    return out


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _parse_aug_scale_from_row(row, meta: Optional[Dict[str, Any]] = None) -> float:
    """
    Returns the offline temporal augmentation factor for one manifest row.

    Priority:
      1) explicit manifest columns: aug_scale or aug_factor
      2) NPZ meta keys: aug_scale or aug_factor
      3) seq pattern: <seq>_x<scale>, e.g. 01-01_x1.4
      4) fallback: 1.0 for real/original sequences
    """
    for col in ("aug_scale", "aug_factor"):
        try:
            if col in row.index:
                v = _safe_float(row[col])
                if np.isfinite(v) and v > 0:
                    return float(v)
        except Exception:
            pass

    if isinstance(meta, dict):
        for key in ("aug_scale", "aug_factor"):
            if key in meta:
                v = _safe_float(meta.get(key))
                if np.isfinite(v) and v > 0:
                    return float(v)

    try:
        seq = str(row.get("seq", ""))
    except Exception:
        seq = ""

    # Match final augmentation suffix: 01-01_x1.4, subjectA_x0.8, etc.
    match = re.search(r"_x([0-9]+(?:\.[0-9]+)?)$", seq)
    if match is not None:
        v = _safe_float(match.group(1))
        if np.isfinite(v) and v > 0:
            return float(v)

    return 1.0


class NPZWindowDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str, window: WindowSpec, max_windows_per_seq: Optional[int] = None):
        df = pd.read_csv(
            manifest_csv,
            dtype={"split": str, "path": str, "seq": str, "subject_id": str},
        )
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

        # allow override from caller (your scripts already pass this)
        if max_windows_per_seq is not None:
            window.max_windows_per_seq = int(max_windows_per_seq)

        self.window = window
        self.index_map: List[Dict[str, Any]] = []

        # 1) First pass: collect fs across sequences to define a single global win_len_ref
        fs_list: List[float] = []
        seq_info: List[Tuple[int, int]] = []  # (row_idx, T)

        for i in range(len(df)):
            path =PROJECT_ROOT / df.loc[i, "path"]
            try:
                with np.load(path, allow_pickle=True) as z:
                    t = z["t"].astype(np.float64)
                    meta = _load_meta(z)
            except Exception:
                continue

            fs = _get_fs_from_meta_or_t(meta, t)
            if not np.isfinite(fs) or fs <= 0:
                continue

            T = int(len(t))
            if T < 3:
                continue

            fs_list.append(float(fs))
            seq_info.append((int(i), T))

        if len(seq_info) == 0:
            raise RuntimeError(f"Our dataset has 0 valid sequences for split={split}. Check manifest/paths.")

        if window.fs_ref is not None and np.isfinite(window.fs_ref) and window.fs_ref > 1.0:
            fs_ref = float(window.fs_ref)
        else:
            fs_ref = float(np.median(np.asarray(fs_list, dtype=np.float64)))

        self.fs_ref = fs_ref
        self.win_len_ref = int(round(window.win_s * fs_ref))
        self.step_len_ref = int(max(1, round(window.stride_s * fs_ref)))

        if self.win_len_ref < 4:
            raise RuntimeError(f"Our computed win_len_ref is too small: {self.win_len_ref}. Check win_s/fs_ref.")

        # 2) Second pass: build windows using fixed reference length
        for (i, T) in seq_info:
            if T < self.win_len_ref:
                continue

            starts = list(range(0, T - self.win_len_ref + 1, self.step_len_ref))

            if window.max_windows_per_seq is not None and len(starts) > window.max_windows_per_seq:
                starts = starts[:window.max_windows_per_seq]

            for s in starts:
                self.index_map.append({"i": int(i), "s": int(s)})

        if len(self.index_map) == 0:
            raise RuntimeError(f"Our dataset has 0 windows for split={split}. Check win_s/stride_s and sequence lengths.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        m = self.index_map[idx]
        i = m["i"]
        s = m["s"]
        L = int(self.win_len_ref)

        path =  PROJECT_ROOT / self.df.loc[i, "path"]
        
        with np.load(path, allow_pickle=True) as z:
            X = z["X"].astype(np.float32)  # [T, C]
            Y = z["Y"].astype(np.float32)  # [T]
            t = z["t"].astype(np.float64)  # [T]
            meta = _load_meta(z)

        fs = _get_fs_from_meta_or_t(meta, t)
        aug_scale = _parse_aug_scale_from_row(self.df.loc[i], meta)
        Xw = X[s:s + L, :]
        Yw = Y[s:s + L]
        tw = t[s:s + L]

        # Force fixed length always (pad/trim)
        Xw = _pad_or_trim_2d(np.ascontiguousarray(Xw, dtype=np.float32), L)
        Yw = _pad_or_trim_1d(np.ascontiguousarray(Yw, dtype=np.float32), L)
        tw = _pad_or_trim_1d(np.ascontiguousarray(tw, dtype=np.float64), L)

        # Clone avoids weird storage issues with collate on some systems
        return (
            torch.from_numpy(Xw).clone(),
            torch.from_numpy(Yw).clone(),
            torch.from_numpy(tw).clone(),
            torch.tensor(fs, dtype=torch.float32),
            torch.tensor(aug_scale, dtype=torch.float32),
        )



class NPZRandomWindowDataset(Dataset):
    """
    Random window sampling per epoch.

    - Dataset length = num_sequences * windows_per_seq_per_epoch
    - Each item maps to (sequence i, random start s)
    - Call set_epoch(epoch) from training loop to change randomness each epoch.

    Uses the same global reference window length (win_len_ref) as NPZWindowDataset.
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        window: WindowSpec,
        windows_per_seq_per_epoch: int = 80,
        seed: int = 123,
    ):
        df = pd.read_csv(
            manifest_csv,
            dtype={"split": str, "path": str, "seq": str, "subject_id": str},
        )
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.window = window
        self.windows_per_seq_per_epoch = int(windows_per_seq_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        # ---- first pass: get fs_ref exactly like your NPZWindowDataset ----
        fs_list: List[float] = []
        seq_info: List[Tuple[int, int]] = []  # (row_idx, T)

        for i in range(len(df)):
            path = PROJECT_ROOT / df.loc[i, "path"]
            try:
                with np.load(path, allow_pickle=True) as z:
                    t = z["t"].astype(np.float64)
                    meta = _load_meta(z)
            except Exception:
                continue

            fs = _get_fs_from_meta_or_t(meta, t)
            if not np.isfinite(fs) or fs <= 0:
                continue

            T = int(len(t))
            if T < 3:
                continue

            fs_list.append(float(fs))
            seq_info.append((int(i), T))

        if len(seq_info) == 0:
            raise RuntimeError(f"Our dataset has 0 valid sequences for split={split}. Check manifest/paths.")

        if window.fs_ref is not None and np.isfinite(window.fs_ref) and window.fs_ref > 1.0:
            fs_ref = float(window.fs_ref)
        else:
            fs_ref = float(np.median(np.asarray(fs_list, dtype=np.float64)))

        self.fs_ref = fs_ref
        self.win_len_ref = int(round(window.win_s * fs_ref))
        if self.win_len_ref < 4:
            raise RuntimeError(f"Our computed win_len_ref is too small: {self.win_len_ref}. Check win_s/fs_ref.")

        # ---- store valid sequences and their max_start ----
        # Only keep sequences with enough length for this window size.
        self.seq_rows: List[int] = []      # row indices in df
        self.seq_max_start: List[int] = [] # max start index for each seq

        for (i, T) in seq_info:
            if T < self.win_len_ref:
                continue
            self.seq_rows.append(int(i))
            self.seq_max_start.append(int(T - self.win_len_ref))

        if len(self.seq_rows) == 0:
            raise RuntimeError(f"Our dataset has 0 usable sequences for split={split} (too short).")

        # small cache to avoid reloading same npz many times (optional but helpful)
        # add fs
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.seq_rows) * self.windows_per_seq_per_epoch

    def _load_npz_cached(self, path: Path):
        key = str(path)
        if key in self._cache:
            return self._cache[key]

        with np.load(path, allow_pickle=True) as z:
            X = z["X"].astype(np.float32)
            Y = z["Y"].astype(np.float32)
            t = z["t"].astype(np.float64)
            meta = _load_meta(z)

        fs = float(_get_fs_from_meta_or_t(meta, t))
        self._cache[key] = (X, Y, t, fs)
        return X, Y, t, fs


    def __getitem__(self, idx):
        # Map idx -> (seq_idx, k_idx)
        seq_idx = idx // self.windows_per_seq_per_epoch
        k_idx = idx % self.windows_per_seq_per_epoch

        row_i = self.seq_rows[seq_idx]
        max_start = self.seq_max_start[seq_idx]
        L = int(self.win_len_ref)

        # Deterministic RNG per (epoch, seq_idx, k_idx)
        rng = np.random.RandomState(self.seed + 100000 * self.epoch + 1000 * seq_idx + k_idx)
        s = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0

        path = PROJECT_ROOT / self.df.loc[row_i, "path"]
        X, Y, t, fs = self._load_npz_cached(path)
        aug_scale = _parse_aug_scale_from_row(self.df.loc[row_i])

        Xw = X[s:s + L, :]
        Yw = Y[s:s + L]
        tw = t[s:s + L]

        # Force fixed length always (pad/trim)
        Xw = _pad_or_trim_2d(np.ascontiguousarray(Xw, dtype=np.float32), L)
        Yw = _pad_or_trim_1d(np.ascontiguousarray(Yw, dtype=np.float32), L)
        tw = _pad_or_trim_1d(np.ascontiguousarray(tw, dtype=np.float64), L)

        return (
            torch.from_numpy(Xw).clone(),
            torch.from_numpy(Yw).clone(),
            torch.from_numpy(tw).clone(),
            torch.tensor(fs, dtype=torch.float32),
            torch.tensor(aug_scale, dtype=torch.float32),
        )
