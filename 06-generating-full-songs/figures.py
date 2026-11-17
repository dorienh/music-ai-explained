"""Draws the article's figures from the saved results.
Run after by_hand.py, vocal_masking.py, generate_long.py and structure.py.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import plt, BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF
IMG = os.path.join(HERE, "..", "..", "images", "06")
if not os.path.isdir(os.path.join(HERE, "..", "..", "images")): IMG = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
save = lambda fig, name: (fig.savefig(os.path.join(IMG, name), dpi=200, bbox_inches="tight"), print("wrote", name))

# ---------- lyrics on the latent grid ----------
d = json.load(open(os.path.join(HERE, "by_hand.json")))
fps = d["fps"]; a, b = int(1.85 * fps), int(7.15 * fps)
fig, axes = plt.subplots(2, 1, figsize=(11, 3.7), sharex=True)
for ax, key, title in ((axes[0], "dr", "Sentence-level (DiffRhythm): each line's phonemes written from its start time"),
                       (axes[1], "jam", "Word-level (JAM): each word's phonemes spread across that word's sung duration")):
    seq = d[key][a:b]
    for i, s in enumerate(seq):
        x = (a + i) / fps
        if s == ".": continue
        col = BLUE if s != "-" else "#cfe0f5"
        ax.add_patch(plt.Rectangle((x, 0), 1 / fps * 0.92, 1, color=col, lw=0))
        if s != "-": ax.text(x + 0.46 / fps, 0.5, s, ha="center", va="center", fontsize=6.5, color="white", rotation=90)
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=20 if key == "jam" else 6)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
for w, s, e in d["words"]:
    axes[1].annotate(w, ((s + e) / 2, 1.02), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=9, color=ORANGE)
axes[1].set_xlim(a / fps, b / fps); axes[1].set_xlabel("Time (s). One cell = one latent frame of 46 ms")
fig.tight_layout(); save(fig, "fig_lyrics_grid.png")

# ---------- words lost in the mix ----------
p = os.path.join(HERE, "vocal_masking.json")
if os.path.exists(p):
    rows = json.load(open(p))
    vars_ = [v for v in [6, 0, -4, -8, -12]]
    m = lambda k, v: 100 * np.mean([r[k] for r in rows if r["var"] == v])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, col, lab in (("mixed", ORANGE, "Mix coded as one token stream"),
                        ("split", BLUE, "Voice and music coded separately"),
                        ("raw", INK2, "Uncoded mix")):
        ys = [m(k, v) for v in vars_]
        ax.plot(vars_, ys, marker="o", color=col, ms=6, lw=2 if k != "raw" else 1.5, ls="-" if k != "raw" else "--")
        ax.annotate(lab, (vars_[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=col)
    ax.axhline(m("mixed", None), color=GRID, lw=1); ax.text(4.5, m("mixed", None) - 0.6, "voice alone through the codec", fontsize=8.5, color=INK2, va="top")
    ax.set_xlim(6.5, -16.5); ax.set_xticks(vars_); ax.set_xticklabels([f"{v:+d}" if v else "0" for v in vars_])
    ax.set_xlabel("Voice level relative to the music (dB)  →  music gets louder"); ax.set_ylabel("Words wrong (WER, %)")
    ax.set_ylim(bottom=0)
    save(fig, "fig_words_lost.png")

# ---------- self-similarity ----------
pa, pb = os.path.join(HERE, "out", "ssm_1.npy"), os.path.join(HERE, "out", "ssm_4.npy")
if os.path.exists(pa) and os.path.exists(pb):
    st = json.load(open(os.path.join(HERE, "structure.json")))
    names = list(st)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    for ax, path, name, label in ((axes[0], pa, names[0], "A real song (fishin, first 90 s)"),
                                  (axes[1], pb, names[3], "MusicGen, 30-second windows chained to 90 s (seed 0)")):
        A = np.load(path); t, S = A[0], A[1:]
        ax.imshow(S, origin="lower", cmap="Blues", vmin=0, vmax=1, extent=[t[0], t[-1], t[0], t[-1]], aspect="equal")
        ax.set_title(f"{label}\nlast third vs first third: {st[name]['last_vs_first']:.2f}", fontsize=10, loc="left")
        n = t[-1] / 3
        ax.add_patch(plt.Rectangle((0, 2 * n), n, n, fill=False, ec=ORANGE, lw=1.5, ls="--"))
        ax.set_xlabel("Time (s)"); ax.grid(False)
    axes[0].set_ylabel("Time (s)")
    fig.tight_layout(); save(fig, "fig_self_similarity.png")
