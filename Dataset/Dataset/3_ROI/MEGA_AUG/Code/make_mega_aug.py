"""
make_mega_aug.py — offline, waveform-preserving augmentation across all 6 datasets
──────────────────────────────────────────────────────────────────────────────────
Reads each per-dataset manifest_split.csv and builds a MEGA augmentation folder:

    <OUT_DIR>/
        manifest_mega_aug.csv     # all originals + augmented train rows
        manifest_mega_orig.csv    # originals only (ablation baseline, no aug)
        aug_npz/                  # resampled .npz files

Split strategy (applied FRESH — ignores existing split column):
  - Subject-level 80/20 split: unique subjects sorted naturally,
    first 80% → train, last 20% → val.
  - All sequences from a given subject land in the SAME split (no leakage).
  - UBFCPhys dataset label is renamed from 'UBFC' to 'UBFCPhys' for disambiguation.

Augmentation rules:
  - Temporal RESAMPLING only — waveform-preserving (pulse shape kept). NO time-flip.
  - In-band guard: factor is applied only if median_HR * factor stays within [BPM_MIN, BPM_MAX].
  - ONLY 'train' rows are augmented. val rows stay original-only.
  - Both manifests use paths relative to PROJECT_ROOT.

After running:
  - Full training  → point MANIFEST_PATH at manifest_mega_aug.csv
  - Ablation study → point MANIFEST_PATH at manifest_mega_orig.csv
"""
from __future__ import annotations
import json
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/media/data/rPPG/Code/GitHub/Project_rPPG")
DATA_ROOT    = PROJECT_ROOT / "Dataset" / "3_ROI"
OUT_DIR      = DATA_ROOT / "MEGA_AUG"

# Each entry: (folder_name, dataset_label_override or None)
# UBFCPhys_RAW has dataset='UBFC' in its manifest — override to 'UBFCPhys'.
DATASETS = [
    ("BH_RAW",        None),
    ("COHFACE_RAW",   None),
    ("PURE_RAW",      None),
    ("TokyoTech_RAW", None),
    ("UBFC_RAW",      None),
    ("UBFCPhys_RAW",  "UBFCPhys"),
]

FACTORS   = [0.55, 0.65, 0.75, 1.20, 1.40, 1.60, 1.80]
BPM_MIN, BPM_MAX = 40.0, 180.0
WINDOW_S  = 8.0       # sliding window length for median HR estimation
STRIDE_S  = 1.0       # sliding window stride
ROI_INDEX = "avg"      # "avg" = average across all ROIs; integer = specific ROI index
TRAIN_RATIO = 0.80     # first 80% of subjects → train, rest → val
# ──────────────────────────────────────────────────────────────────────────────


# ── SUBJECT EXTRACTION ────────────────────────────────────────────────────────
# The subject_id column in some manifests includes session/task info.
# These functions extract the TRUE subject identifier per dataset.

def _extract_subject_bh(subject_id: str) -> str:
    """BH: '0_0' → '0' (first number is the subject, second is session)."""
    return subject_id.split("_")[0]

def _extract_subject_cohface(subject_id: str) -> str:
    """COHFACE: 'Subj_10_0' → 'Subj_10' (last part is session index)."""
    parts = subject_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else subject_id

def _extract_subject_pure(subject_id: str) -> str:
    """PURE: '01' → '01' (already the true subject)."""
    return subject_id

def _extract_subject_tokyotech(subject_id: str) -> str:
    """TokyoTech: 'Subj_01_Exercise_Frag4' → 'Subj_01'."""
    parts = subject_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else subject_id

def _extract_subject_ubfc(subject_id: str) -> str:
    """UBFC: 'vid_1' → 'vid_1' (each video = one subject)."""
    return subject_id

def _extract_subject_ubfcphys(subject_id: str) -> str:
    """UBFCPhys: 's10_T1' → 's10' (part before _T is subject)."""
    m = re.match(r"(s\d+)", subject_id)
    return m.group(1) if m else subject_id

# Map from the EFFECTIVE dataset label (after override) to extractor
SUBJECT_EXTRACTORS = {
    "BH":        _extract_subject_bh,
    "COHFACE":   _extract_subject_cohface,
    "PURE":      _extract_subject_pure,
    "TokyoTech": _extract_subject_tokyotech,
    "UBFC":      _extract_subject_ubfc,
    "UBFCPhys":  _extract_subject_ubfcphys,
}


def natural_sort_key(s: str):
    """Sort key that orders numeric parts numerically: 'vid_2' < 'vid_10'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def compute_subject_split(subjects: list[str], train_ratio: float) -> dict[str, str]:
    """
    Sort subjects naturally, assign first train_ratio fraction to 'train', rest to 'val'.
    Returns dict: {subject → 'train' or 'val'}.
    """
    sorted_subjs = sorted(set(subjects), key=natural_sort_key)
    n_total = len(sorted_subjs)
    n_train = math.ceil(n_total * train_ratio)
    # Ensure at least 1 subject in val
    if n_train >= n_total and n_total > 1:
        n_train = n_total - 1
    split_map = {}
    for i, subj in enumerate(sorted_subjs):
        split_map[subj] = "train" if i < n_train else "val"
    return split_map


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_meta(m):
    """Extract metadata dict from npz 'meta' field (handles bytes/str/ndarray)."""
    if isinstance(m, np.ndarray):
        m = m.item()
    if isinstance(m, (bytes, bytearray)):
        m = m.decode("utf-8", "ignore")
    if isinstance(m, str):
        try:
            return json.loads(m)
        except Exception:
            return {}
    return m if isinstance(m, dict) else {}


def resolve(path_value, manifest_path):
    """Resolve a possibly-relative path against manifest parent and ancestors."""
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    for parent in [manifest_path.parent, Path.cwd(), *manifest_path.resolve().parents]:
        candidate = parent / p
        if candidate.exists():
            return candidate
    return manifest_path.parent / p


def infer_fs(meta, t):
    """Infer sampling rate from metadata fps field, falling back to timestamp diffs."""
    fs = float(meta.get("fps", np.nan))
    if np.isfinite(fs) and fs > 1:
        return fs
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else 30.0


def fft_hr(sig, fs):
    """Peak heart rate (BPM) from FFT of a single BVP segment."""
    sig = np.asarray(sig, float) - np.mean(sig)
    power = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs) * 60
    mask = (freqs >= BPM_MIN) & (freqs <= BPM_MAX)
    if not mask.any():
        return np.nan
    return float(freqs[mask][np.argmax(power[mask])])


def median_hr(Y, fs):
    """Robust median HR estimate via sliding-window FFT."""
    win = int(round(WINDOW_S * fs))
    step = int(round(STRIDE_S * fs))
    if len(Y) < win:
        return fft_hr(Y, fs)
    hrs = [fft_hr(Y[s:s + win], fs) for s in range(0, len(Y) - win + 1, step)]
    hrs = [h for h in hrs if np.isfinite(h)]
    return float(np.median(hrs)) if hrs else np.nan


def resample_factor(arr, factor, min_samples):
    """
    Resample array by `factor`: new_T = round(T / factor).
    factor < 1 → more samples → slower HR.
    factor > 1 → fewer samples → faster HR.
    Returns None if result is shorter than min_samples.
    """
    T = arr.shape[0]
    new_T = int(round(T / factor))
    if new_T < min_samples:
        return None
    src = np.linspace(0, T - 1, new_T)
    base = np.arange(T)
    if arr.ndim == 1:
        return np.interp(src, base, arr).astype(arr.dtype)
    cols = [np.interp(src, base, arr[:, c]) for c in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(arr.dtype)


def make_relative(abs_path, root):
    """Convert absolute path to relative w.r.t. root; fallback to absolute string."""
    try:
        return str(Path(abs_path).relative_to(root))
    except ValueError:
        return str(abs_path)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    aug_dir = OUT_DIR / "aug_npz"
    aug_dir.mkdir(parents=True, exist_ok=True)

    all_orig_rows = []   # originals only (ablation manifest)
    all_rows      = []   # originals + augmented (full manifest)

    n_orig = 0
    n_aug  = 0
    n_skip_band  = 0
    n_skip_short = 0
    n_skip_load  = 0

    # minimum sample count after resampling (WINDOW_S seconds at ~30 fps as lower bound)
    min_samples = int(WINDOW_S * 30)

    for folder_name, ds_override in DATASETS:
        manifest_path = DATA_ROOT / folder_name / "manifest_split.csv"
        if not manifest_path.exists():
            print(f"[SKIP] manifest not found: {manifest_path}")
            continue

        man = pd.read_csv(manifest_path, dtype=str)
        for col in ("split", "seq", "subject_id", "path", "dataset"):
            if col not in man.columns:
                man[col] = "unknown"

        # --- Determine effective dataset label ---
        ds_label = ds_override if ds_override else man["dataset"].iloc[0]

        # --- Get subject extractor ---
        extractor = SUBJECT_EXTRACTORS.get(ds_label)
        if extractor is None:
            print(f"[ERROR] No subject extractor defined for dataset '{ds_label}'")
            continue

        # --- Extract true subject IDs and compute 80/20 split ---
        true_subjects = [extractor(sid) for sid in man["subject_id"]]
        unique_subjects = list(set(true_subjects))
        split_map = compute_subject_split(unique_subjects, TRAIN_RATIO)

        # Print the split assignment
        train_subjs = sorted([s for s, v in split_map.items() if v == "train"], key=natural_sort_key)
        val_subjs   = sorted([s for s, v in split_map.items() if v == "val"], key=natural_sort_key)

        print(f"\n{'─'*70}")
        print(f"Processing: {folder_name}  (label={ds_label}, {len(man)} rows, {len(unique_subjects)} subjects)")
        print(f"  Train subjects ({len(train_subjs):2d}): {train_subjs}")
        print(f"  Val   subjects ({len(val_subjs):2d}): {val_subjs}")
        print(f"{'─'*70}")

        ds_orig = 0
        ds_aug  = 0

        for _, row in man.iterrows():
            src_path = resolve(row["path"], manifest_path)
            orig = dict(row)

            # --- Fix dataset label ---
            orig["dataset"] = ds_label

            # --- Assign split from subject-level split map (replaces old split) ---
            true_subj = extractor(row["subject_id"])
            orig["split"] = split_map[true_subj]

            # --- Make path relative to PROJECT_ROOT ---
            orig["path"] = make_relative(src_path, PROJECT_ROOT)

            all_orig_rows.append(dict(orig))
            all_rows.append(dict(orig))
            n_orig += 1
            ds_orig += 1

            # --- Augment only train rows ---
            if orig["split"] != "train":
                continue
            if not src_path.exists():
                n_skip_load += 1
                continue

            try:
                with np.load(src_path, allow_pickle=True) as z:
                    X = z["X"].astype(np.float32)
                    Y = z["Y"].astype(np.float32)
                    t = z["t"].astype(np.float64)
                    meta = load_meta(z["meta"])
            except Exception as e:
                print(f"  [WARN] load failed: {src_path.name} — {e}")
                n_skip_load += 1
                continue

            fs  = infer_fs(meta, t)
            mhr = median_hr(Y, fs)
            if not np.isfinite(mhr):
                continue

            for f in FACTORS:
                # In-band guard
                if mhr * f > BPM_MAX or mhr * f < BPM_MIN:
                    n_skip_band += 1
                    continue

                Xa = resample_factor(X, f, min_samples)
                Ya = resample_factor(Y, f, min_samples)
                if Xa is None or Ya is None:
                    n_skip_short += 1
                    continue

                new_T = Xa.shape[0]
                ta = (np.arange(new_T) / fs).astype(np.float64)

                # Build augmented metadata
                meta_a = dict(meta) if isinstance(meta, dict) else {}
                meta_a["fps"] = fs
                meta_a["aug_factor"] = f
                meta_a["seq"] = f"{row['seq']}_x{f}"

                # Save augmented npz
                fname = (
                    f"{ds_label}_{row['subject_id']}_{row['seq']}_x{f}.npz"
                    .replace("/", "_")
                    .replace("'", "")
                )
                out_npz = aug_dir / fname
                np.savez(out_npz, X=Xa, Y=Ya, t=ta, meta=json.dumps(meta_a))

                # Build augmented manifest row
                ar = dict(orig)
                ar["path"] = make_relative(out_npz, PROJECT_ROOT)
                ar["seq"]  = f"{row['seq']}_x{f}"
                ar["T"]    = str(new_T)
                ar["t_end"] = f"{ta[-1]:.6f}"
                all_rows.append(ar)
                n_aug += 1
                ds_aug += 1

        print(f"  originals: {ds_orig}   augmented: {ds_aug}")

    # --- Determine column order from the first manifest ---
    first_manifest = DATA_ROOT / DATASETS[0][0] / "manifest_split.csv"
    col_order = list(pd.read_csv(first_manifest, dtype=str, nrows=0).columns)

    # --- Write full manifest (originals + augmented) ---
    df_aug = pd.DataFrame(all_rows)
    df_aug = df_aug[[c for c in col_order if c in df_aug.columns]]
    aug_path = OUT_DIR / "manifest_mega_aug.csv"
    df_aug.to_csv(aug_path, index=False)

    # --- Write ablation manifest (originals only) ---
    df_orig = pd.DataFrame(all_orig_rows)
    df_orig = df_orig[[c for c in col_order if c in df_orig.columns]]
    orig_path = OUT_DIR / "manifest_mega_orig.csv"
    df_orig.to_csv(orig_path, index=False)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"MEGA AUG — DONE")
    print(f"{'='*60}")
    print(f"Original rows total      : {n_orig}")
    print(f"Augmented files created  : {n_aug}")
    print(f"Skipped (out-of-band)    : {n_skip_band}")
    print(f"Skipped (too short)      : {n_skip_short}")
    print(f"Skipped (load failed)    : {n_skip_load}")
    print(f"")
    print(f"manifest_mega_aug.csv    : {len(df_aug)} rows  → {aug_path}")
    print(f"manifest_mega_orig.csv   : {len(df_orig)} rows → {orig_path}")
    print(f"Aug npz directory        : {aug_dir}")

    # --- Per-dataset breakdown ---
    print(f"\n{'─'*60}")
    print("Per-dataset breakdown (mega_aug):")
    for ds in sorted(df_aug["dataset"].unique()):
        sub = df_aug[df_aug["dataset"] == ds]
        n_orig_ds = (~sub["seq"].str.contains("_x", na=False)).sum()
        n_aug_ds  = sub["seq"].str.contains("_x", na=False).sum()
        n_tr = (sub["split"] == "train").sum()
        n_va = (sub["split"] == "val").sum()
        print(f"  {ds:12s}  orig={n_orig_ds:4d}  aug={n_aug_ds:5d}  train={n_tr:5d}  val={n_va:4d}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
