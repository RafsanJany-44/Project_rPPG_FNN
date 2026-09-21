"""
analyze_hr_distribution.py — compute median HR per sequence and save to CSV
─────────────────────────────────────────────────────────────────────────────
Reads a manifest, loads each .npz, computes median HR from the BVP signal,
and writes a CSV with columns: dataset, seq, subject_id, split, is_aug, median_hr, fs.

Run for one or all datasets. Output CSV can be used for balance analysis.

Usage:
    python analyze_hr_distribution.py                    # all datasets (mega manifest)
    python analyze_hr_distribution.py --manifest /path/to/manifest_BH.csv   # single dataset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/media/data/rPPG/Code/GitHub/Project_rPPG")
DEFAULT_MANIFEST = PROJECT_ROOT / "Dataset" / "3_ROI" / "MEGA_AUG" / "manifest_mega_aug.csv"
OUTPUT_CSV = Path("hr_distribution.csv")

BPM_MIN, BPM_MAX = 40.0, 180.0
WINDOW_S = 8.0
STRIDE_S = 1.0
# ──────────────────────────────────────────────────────────────────────────────


def load_meta(m):
    if isinstance(m, np.ndarray): m = m.item()
    if isinstance(m, (bytes, bytearray)): m = m.decode("utf-8", "ignore")
    if isinstance(m, str):
        try: return json.loads(m)
        except Exception: return {}
    return m if isinstance(m, dict) else {}


def infer_fs(meta, t):
    fs = float(meta.get("fps", np.nan))
    if np.isfinite(fs) and fs > 1: return fs
    dt = np.diff(t); dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else 30.0


def fft_hr(sig, fs):
    sig = np.asarray(sig, float) - np.mean(sig)
    if len(sig) < 16: return np.nan
    power = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs) * 60
    mask = (freqs >= BPM_MIN) & (freqs <= BPM_MAX)
    if not mask.any(): return np.nan
    return float(freqs[mask][np.argmax(power[mask])])


def median_hr(Y, fs):
    win = int(round(WINDOW_S * fs))
    step = int(round(STRIDE_S * fs))
    if len(Y) < win:
        return fft_hr(Y, fs)
    hrs = [fft_hr(Y[s:s + win], fs) for s in range(0, len(Y) - win + 1, step)]
    hrs = [h for h in hrs if np.isfinite(h)]
    return float(np.median(hrs)) if hrs else np.nan


def resolve_path(path_value, manifest_path):
    p = Path(str(path_value))
    if p.is_absolute() and p.exists(): return p
    # Try relative to PROJECT_ROOT
    c = PROJECT_ROOT / p
    if c.exists(): return c
    # Try relative to manifest
    c = manifest_path.parent / p
    if c.exists(): return c
    return PROJECT_ROOT / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    print(f"Manifest : {manifest_path}")
    print(f"Output   : {output_path}")

    man = pd.read_csv(manifest_path, dtype=str)
    total = len(man)
    results = []

    for i, (_, row) in enumerate(man.iterrows()):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Processing {i+1}/{total}...", end="\r")

        src = resolve_path(row["path"], manifest_path)
        is_aug = "_x" in str(row.get("seq", ""))

        if not src.exists():
            results.append({
                "dataset": row.get("dataset", ""),
                "seq": row.get("seq", ""),
                "subject_id": row.get("subject_id", ""),
                "split": row.get("split", ""),
                "is_aug": is_aug,
                "median_hr": np.nan,
                "fs": np.nan,
            })
            continue

        try:
            with np.load(src, allow_pickle=True) as z:
                Y = z["Y"].astype(np.float32)
                t = z["t"].astype(np.float64)
                meta = load_meta(z["meta"])
            fs = infer_fs(meta, t)
            mhr = median_hr(Y, fs)
        except Exception:
            mhr = np.nan
            fs = np.nan

        results.append({
            "dataset": row.get("dataset", ""),
            "seq": row.get("seq", ""),
            "subject_id": row.get("subject_id", ""),
            "split": row.get("split", ""),
            "is_aug": is_aug,
            "median_hr": round(mhr, 2) if np.isfinite(mhr) else np.nan,
            "fs": round(fs, 2) if np.isfinite(fs) else np.nan,
        })

    print()
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Quick summary
    print(f"\n{'─'*70}")
    print("Quick HR summary per dataset + split:")
    print(f"{'─'*70}")
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        for split in ["train", "val"]:
            sp = sub[sub["split"] == split]
            hrs = sp["median_hr"].dropna()
            if len(hrs) == 0: continue
            orig_hrs = sp[~sp["is_aug"]]["median_hr"].dropna()
            aug_hrs  = sp[sp["is_aug"]]["median_hr"].dropna()
            print(f"  {ds:12s} {split:5s}  "
                  f"all: n={len(hrs):4d} mean={hrs.mean():5.1f} std={hrs.std():5.1f} "
                  f"[{hrs.min():5.1f}–{hrs.max():5.1f}]  "
                  f"orig: n={len(orig_hrs):3d}  aug: n={len(aug_hrs):4d}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
