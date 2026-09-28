"""
compare_aug.py — side-by-side viewer: original vs augmented copies of one subject
─────────────────────────────────────────────────────────────────────────────
Reads the AUG manifest, finds all rows for a chosen seq (original + _x* copies),
and builds ONE interactive Plotly HTML comparing them:
  • time-domain BVP waveform (shape preserved check)
  • RGB green channel (the resampled input)
  • frequency spectrum (HR shifted up by the factor)
Set CONFIG and run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ────────────────────────────────────────────────────────────────────
AUG_MANIFEST = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/PURE-x-UBFC-x-Tokoyo-AUG/manifest_split.csv")
SAVE_DIR     = Path("./aug_compare_out")
SEQ          = "'01-01"     # base seq name (without _x..); script finds all its copies
ROI_INDEX    = "avg"
BPM_MIN, BPM_MAX = 40.0, 180.0
SHOW_SECONDS = 6.0         # how many seconds of waveform to plot
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

def spectrum(sig, fs):
    sig=np.asarray(sig,float)-np.mean(sig)
    p=np.abs(np.fft.rfft(sig))**2
    f=np.fft.rfftfreq(len(sig),d=1/fs)*60
    m=(f>=BPM_MIN)&(f<=BPM_MAX)
    return f[m], p[m]/(p[m].max()+1e-12)

def peak(sig,fs):
    f,p=spectrum(sig,fs); return float(f[np.argmax(p)]) if len(p) else np.nan


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    man=pd.read_csv(AUG_MANIFEST, dtype=str)
    # match the base seq and all its _x.. variants
    rows=man[(man.seq==SEQ) | (man.seq.str.startswith(f"{SEQ}_x"))]
    if len(rows)==0:
        raise SystemExit(f"no rows for seq={SEQ} (or {SEQ}_x*) in {AUG_MANIFEST}")
    # order: original first, then by factor
    def order_key(s):
        if s==SEQ: return 0.0
        try: return float(s.split("_x")[-1])
        except: return 99.0
    rows=rows.assign(_k=rows.seq.map(order_key)).sort_values("_k")

    fig=make_subplots(rows=3, cols=1, vertical_spacing=0.08,
        subplot_titles=("BVP waveform (shape should be preserved, just faster)",
                        "Green channel input (resampled)",
                        "Spectrum (HR peak shifts UP with factor)"))
    colors=["#111111","#1D6FB8","#2E7D52","#C0532B","#7B3FA0","#C9851A"]

    for i,(_,r) in enumerate(rows.iterrows()):
        p=resolve(r["path"], AUG_MANIFEST)
        if not p.exists(): 
            print("missing:",p); continue
        with np.load(p, allow_pickle=True) as z:
            X=z["X"].astype(np.float32); Y=z["Y"].astype(np.float32)
            t=z["t"].astype(np.float64); meta=load_meta(z["meta"])
        fs=infer_fs(meta,t); Xr=select_roi(X)
        G=Xr[:,1]
        hr=peak(Y,fs)
        col=colors[i%len(colors)]
        label=f"{r['seq']}  (HR≈{hr:.0f}, fs={fs:.1f}, N={len(Y)})"
        n=int(SHOW_SECONDS*fs)
        tt=np.arange(min(n,len(Y)))/fs
        z1=lambda s:(s-np.mean(s))/(np.std(s)+1e-8)

        fig.add_trace(go.Scatter(x=tt,y=z1(Y[:len(tt)]),name=label,legendgroup=label,
            line=dict(color=col)),row=1,col=1)
        fig.add_trace(go.Scatter(x=tt,y=z1(G[:len(tt)]),name=label,legendgroup=label,
            showlegend=False,line=dict(color=col)),row=2,col=1)
        f,ps=spectrum(Y,fs)
        fig.add_trace(go.Scatter(x=f,y=ps,name=label,legendgroup=label,
            showlegend=False,line=dict(color=col)),row=3,col=1)

    fig.update_xaxes(title_text="time (s)",row=1,col=1)
    fig.update_xaxes(title_text="time (s)",row=2,col=1)
    fig.update_xaxes(title_text="BPM",row=3,col=1)
    fig.update_layout(template="plotly_white",height=950,width=1300,
        title=f"<b>Augmentation comparison — {SEQ}</b>  (original vs resampled copies)",
        legend=dict(orientation="h",y=1.06))
    out=SAVE_DIR/f"AUG_COMPARE_{SEQ}.html"
    fig.write_html(str(out), include_plotlyjs=True)
    print(f"wrote {out}")
    print("\nVariants shown:")
    for _,r in rows.iterrows():
        print(f"  {r['seq']}")


if __name__ == "__main__":
    main()