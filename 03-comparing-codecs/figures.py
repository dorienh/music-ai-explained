"""Figures for 'I Ran Five Neural Audio Codecs on the Same Music'. Reads results.json (written by codec_benchmark.py)."""
import os, sys, json; sys.path.insert(0, ".."); os.makedirs("out", exist_ok=True)
import numpy as np
from style import *

rows = json.load(open("results.json")); music = ("drums", "strings", "vocals")
sysn = list(dict.fromkeys(r["system"] for r in rows))
avg = lambda s, k: float(np.mean([r[k] for r in rows if r["system"] == s and r["clip"] in music]))
D = {s: dict(kbps=avg(s, "kbps"), mel=avg(s, "mel44"), sdr=avg(s, "si_sdr")) for s in sysn}
classic = lambda s: s.startswith(("Opus", "MP3"))
short = {"EnCodec 24k, 2 cb": "EnCodec 24k ×2", "EnCodec 24k, 8 cb": "EnCodec 24k ×8", "EnCodec 24k, 32 cb": "EnCodec 24k ×32", "EnCodec 32k (MusicGen)": "EnCodec 32k",
         "DAC 44k, 3 cb": "DAC ×3", "DAC 44k, 9 cb": "DAC ×9", "SNAC 44k": "SNAC 44k", "SNAC 24k (speech)": "SNAC 24k", "Mimi, 8 cb": "Mimi",
         "Opus 6k": "Opus 6k", "Opus 12k": "Opus 12k", "MP3 32k": "MP3 32k", "MP3 128k": "MP3 128k"}

def scatter(xk, yk, xlabel, ylabel, fname, logx, off):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for fam, col, lab in [(False, BLUE, "Neural codecs"), (True, ORANGE, "Opus and MP3")]:
        S = [s for s in sysn if classic(s) == fam]
        ax.scatter([D[s][xk] for s in S], [D[s][yk] for s in S], s=80, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=lab)
    for s in sysn:
        dx, dy, ha = off.get(s, (8, 4, "left"))
        ax.annotate(short[s], (D[s][xk], D[s][yk]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9.5, color=INK)
    if logx: ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(f"out/{fname}", dpi=200); plt.close(fig)

scatter("kbps", "mel", "Bitrate (kbit/s)", "Mel distance on music, full range (lower is better)", "fig_bitrate_mel.png", True,
        {"Mimi, 8 cb": (8, 6, "left"), "SNAC 24k (speech)": (0, -18, "center"), "EnCodec 24k, 2 cb": (8, -12, "left"), "SNAC 44k": (8, -12, "left"), "DAC 44k, 3 cb": (-8, 6, "right"),
         "EnCodec 24k, 8 cb": (8, 4, "left"), "Opus 12k": (8, 4, "left")})
scatter("sdr", "mel", "SI-SDR on music, in dB (higher is better)", "Mel distance on music, full range (lower is better)", "fig_metrics_disagree.png", False,
        {"Mimi, 8 cb": (8, 6, "left"), "EnCodec 24k, 2 cb": (-6, -16, "right"), "SNAC 44k": (0, -18, "center"), "DAC 44k, 3 cb": (-8, 6, "right"), "EnCodec 24k, 32 cb": (10, -4, "left"),
         "EnCodec 24k, 8 cb": (-4, 10, "center"), "EnCodec 32k (MusicGen)": (-8, -14, "right")})
