"""
make_offline_aug.py — offline, waveform-preserving augmentation set
─────────────────────────────────────────────────────────────────────────────
Reads the ORIGINAL manifest_split.csv and builds a NEW self-contained folder:

    <OUT_DIR>/
        manifest_split.csv        # same format/columns as the original
        aug_npz/                  # resampled .npz files (new high-HR copies)

Rules :
  • Temporal RESAMPLING only — waveform-preserving (pulse shape kept). NO time-flip.
  • Clearly defined factors (speed-up only -> fills the starved high-HR zone).
  • In-band guard: a factor is used for a sequence only if median_HR*factor stays <= BPM_MAX.
  • ONLY 'train' rows are augmented. val/test stay original-only.
  • New manifest references originals (as absolute paths) + augmented copies, so
    the existing training workflow reads it with ZERO code changes.

After running: point the trainer's MANIFEST_PATH at <OUT_DIR>/manifest_split.csv
and REMOVE the runtime frequency_scale_temporal_aug call.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ORIG_MANIFEST = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/PURE-x-UBFC-x-Tokoyo/manifest_split.csv")
OUT_DIR       = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/PURE-x-UBFC-x-Tokoyo-AUG")
# Paths in the new manifest are written RELATIVE to this root (e.g. "Dataset/3_ROI/...").
PROJECT_ROOT  = Path("/media/data/rPPG/Code/GitHub/Project_rPPG")

#FACTORS   = [0.80, 1.20, 1.40, 1.60, 1.80]   # speed-up factors (1.0 original is referenced, not re-saved)
FACTORS = [0.55, 0.65, 0.75, 1.20, 1.40, 1.60, 1.80]
BPM_MIN, BPM_MAX = 40.0, 180.0
WINDOW_S  = 8.0                  # only used to estimate sequence HR for the in-band guard
STRIDE_S  = 1.0
ROI_INDEX = "avg"
TRAIN_SPLIT_NAMES = {"train"}    # which split values get augmented
# ──────────────────────────────────────────────────────────────────────────────


def load_meta(m):
    if isinstance(m,np.ndarray): m=m.item()
    if isinstance(m,(bytes,bytearray)): m=m.decode("utf-8","ignore")
    if isinstance(m,str):
        try: return json.loads(m)
        except Exception: return {}
    return m if isinstance(m,dict) else {}

def resolve(path_value, manifest):
    p=Path(str(path_value))
    if p.is_absolute(): return p
    for parent in [manifest.parent, Path.cwd(), *manifest.resolve().parents]:
        c=parent/p
        if c.exists(): return c
    return manifest.parent/p

def infer_fs(meta,t):
    fs=float(meta.get("fps",np.nan))
    if np.isfinite(fs) and fs>1: return fs
    dt=np.diff(t); dt=dt[dt>0]
    return float(1.0/np.median(dt)) if dt.size else 30.0

def select_roi(x):
    if ROI_INDEX=="avg":
        T,C=x.shape; return x.reshape(T,C//3,3).mean(axis=1)
    i=int(ROI_INDEX); return x[:, i*3:i*3+3]

def fft_hr(sig, fs):
    sig=np.asarray(sig,float)-np.mean(sig)
    p=np.abs(np.fft.rfft(sig))**2
    f=np.fft.rfftfreq(len(sig),d=1/fs)*60
    m=(f>=BPM_MIN)&(f<=BPM_MAX)
    if not m.any(): return np.nan
    return float(f[m][np.argmax(p[m])])

def median_hr(Y, fs):
    win=int(round(WINDOW_S*fs)); step=int(round(STRIDE_S*fs))
    if len(Y)<win: return fft_hr(Y,fs)
    hrs=[fft_hr(Y[s:s+win],fs) for s in range(0,len(Y)-win+1,step)]
    hrs=[h for h in hrs if np.isfinite(h)]
    return float(np.median(hrs)) if hrs else np.nan

def resample_factor(arr, factor):
    """Speed up by `factor`: resample to fewer samples at the SAME fs.
       arr: [T] or [T,C]. Returns waveform-preserving, HR-multiplied-by-factor signal."""
    T=arr.shape[0]
    # new_T=int(round(T/factor))
    # if new_T<8: return None
    new_T=int(round(T/factor))
    if new_T < int(WINDOW_S * 30): return None
    src=np.linspace(0, T-1, new_T)
    base=np.arange(T)
    if arr.ndim==1:
        return np.interp(src, base, arr).astype(arr.dtype)
    cols=[np.interp(src, base, arr[:,c]) for c in range(arr.shape[1])]
    return np.stack(cols, axis=1).astype(arr.dtype)


def main():
    aug_dir = OUT_DIR/"aug_npz"
    aug_dir.mkdir(parents=True, exist_ok=True)

    man=pd.read_csv(ORIG_MANIFEST, dtype=str)
    for col in ("split","seq","subject_id","path"):
        if col not in man.columns: man[col]="unknown"

    new_rows=[]
    n_orig=0; n_aug=0; n_skip_band=0

    for _,row in man.iterrows():
        # 1) copy the ORIGINAL row, with path resolved to ABSOLUTE so it works from OUT_DIR
        src_path=resolve(row["path"], ORIG_MANIFEST)   # absolute, for loading
        orig=dict(row)
        try:
            orig["path"]=str(src_path.relative_to(PROJECT_ROOT))
        except ValueError:
            orig["path"]=str(row["path"])   # fallback: keep original manifest value
        new_rows.append(orig); n_orig+=1

        # 2) augment only TRAIN rows
        if str(row["split"]).lower() not in TRAIN_SPLIT_NAMES:
            continue
        if not src_path.exists():
            continue
        try:
            with np.load(src_path, allow_pickle=True) as z:
                X=z["X"].astype(np.float32); Y=z["Y"].astype(np.float32)
                t=z["t"].astype(np.float64); meta=load_meta(z["meta"])
        except Exception:
            continue
        fs=infer_fs(meta,t)
        mhr=median_hr(Y, fs)
        if not np.isfinite(mhr):
            continue

        for f in FACTORS:
            # in-band guard: skip if the sped-up HR would exit the band
            if mhr*f > BPM_MAX:
                n_skip_band+=1
                continue
            if mhr*f < BPM_MIN:          # <-- add this
                n_skip_band+=1
                continue
            Xa=resample_factor(X, f)        # resample ALL channels (full [T,C])
            Ya=resample_factor(Y, f)
            if Xa is None or Ya is None:
                continue
            new_T=Xa.shape[0]
            ta=(np.arange(new_T)/fs).astype(np.float64)   # same fs, fewer samples -> higher HR
            meta_a=dict(meta) if isinstance(meta,dict) else {}
            meta_a["fps"]=fs
            meta_a["aug_factor"]=f
            meta_a["seq"]=f"{row['seq']}_x{f}"

            fname=f"{row['dataset']}_{row['subject_id']}_{row['seq']}_x{f}.npz".replace("/","_")
            out_npz=aug_dir/fname
            np.savez(out_npz, X=Xa, Y=Ya, t=ta, meta=json.dumps(meta_a))

            ar=dict(row)
            try:
                ar["path"]=str(out_npz.relative_to(PROJECT_ROOT))
            except ValueError:
                ar["path"]=str(out_npz)   # fallback: absolute if outside root
            ar["seq"]=f"{row['seq']}_x{f}"
            new_rows.append(ar); n_aug+=1

    out_man=pd.DataFrame(new_rows)
    # keep the same column order as the original manifest
    out_man=out_man[[c for c in man.columns]]
    out_path=OUT_DIR/"manifest_split.csv"
    out_man.to_csv(out_path, index=False)

    print(f"Original rows referenced : {n_orig}")
    print(f"Augmented files created  : {n_aug}")
    print(f"Skipped (out-of-band)    : {n_skip_band}")
    print(f"New manifest rows total  : {len(out_man)}")
    print(f"\nNew folder : {OUT_DIR}")
    print(f"Manifest   : {out_path}")
    print(f"Aug npz in : {aug_dir}")
    print("\nNext: point trainer MANIFEST_PATH at the new manifest, remove runtime aug.")


if __name__ == "__main__":
    main()