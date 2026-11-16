"""Figures for 'Neural Audio Codecs Up Close'. Numbers come from toy_rvq.py and codebook_usage_and_vq_layer.py."""
import os, sys; sys.path.insert(0, ".."); os.makedirs("out", exist_ok=True)
import numpy as np
from style import *

# 1. vector quantization in two dimensions
C = np.array([[1, 1], [1, 3], [3, 1], [3, 3]]); x = np.array([1.4, 2.6])
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.axvline(2, color=INK2, lw=1, ls=(0, (4, 3))); ax.axhline(2, color=INK2, lw=1, ls=(0, (4, 3)))
ax.fill_between([0, 2], 2, 4, color=BLUE, alpha=0.10, lw=0)
ax.scatter(C[:, 0], C[:, 1], s=220, color=BLUE, zorder=3)
for i, c in enumerate(C): ax.annotate(f"entry {i}", c, xytext=(0, 16 if i == 1 else -24), textcoords="offset points", ha="center", color=INK)
ax.annotate("", xy=C[1], xytext=x, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2, shrinkA=6, shrinkB=9))
ax.scatter(*x, s=90, color=ORANGE, zorder=4)
ax.annotate("frame (1.4, 2.6)", x, xytext=(10, -4), textcoords="offset points", color=INK)
ax.annotate("stored: index 1 (2 bits)", (0.12, 3.75), color=INK2, fontsize=10)
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal"); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(False)
fig.tight_layout(); fig.savefig("out/fig_vq2d.png", dpi=200); plt.close(fig)

# 2. residual vector quantization as repeated zooming
fig, axs = plt.subplots(3, 1, figsize=(8, 4.2))
rows = [("Codebook 1", [0, 4, 8, 12], 8, (-1, 13)), ("Codebook 2: what is left after 8", [6.5, 7.5, 8.5, 9.5], 7.5, (6, 10)), ("Codebook 3: what is left after 7.5", [7.2, 7.4, 7.6, 7.8], 7.2, (7.1, 7.9))]
for ax, (title, ticks, pick, lim) in zip(axs, rows):
    ax.hlines(0, *lim, color=INK2, lw=1)
    ax.scatter(ticks, [0]*4, s=120, color=[BLUE if t == pick else SURF for t in ticks], edgecolor=BLUE, linewidth=2, zorder=3)
    for t in ticks: ax.annotate(f"{t:g}", (t, 0), xytext=(0, -20), textcoords="offset points", ha="center", color=INK2, fontsize=10)
    ax.scatter([7.26], [0], s=70, color=ORANGE, zorder=4, marker="D")
    ax.annotate("7.26", (7.26, 0), xytext=(0, 11), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_title(title, loc="left", fontsize=11); ax.set_xlim(*lim); ax.set_ylim(-1, 1); ax.axis("off")
fig.text(0.99, 0.02, "stored: 2, 1, 0  →  6 bits      decoded: 8 − 0.5 − 0.3 = 7.2", ha="right", color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("out/fig_rvq_zoom.png", dpi=200); plt.close(fig)

# 3. one codebook against RVQ on unseen music (output of toy_rvq.py)
single = {4: 0.592, 8: 0.503, 10: 0.469}
rvq = {4: 0.610, 8: 0.497, 12: 0.423, 16: 0.372, 20: 0.334, 24: 0.302, 28: 0.276, 32: 0.252}
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(list(rvq), list(rvq.values()), "-o", color=BLUE, ms=8, mec=SURF, mew=2, label="RVQ, stages of 16 entries")
ax.plot(list(single), list(single.values()), "-o", color=ORANGE, ms=8, mec=SURF, mew=2, label="One codebook")
ax.annotate("1,024 entries", (10, 0.469), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=10)
ax.annotate("48 stored vectors", (12, 0.423), xytext=(8, 4), textcoords="offset points", color=INK, fontsize=10)
ax.annotate("128 stored vectors", (32, 0.252), xytext=(-8, 10), textcoords="offset points", ha="right", color=INK, fontsize=10)
ax.set_xticks(range(4, 33, 4)); ax.set_ylim(0, 0.7); ax.set_xlabel("Bits per frame"); ax.set_ylabel("Error on unseen music (lower is better)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("out/fig_rvq_error.png", dpi=200); plt.close(fig)

# 4. codebook usage (output of codebook_usage_and_vq_layer.py)
enc = [8.57, 9.06, 9.17, 9.30, 9.35, 9.37, 9.36, 9.34, 9.39]
dac = [8.71, 9.86, 9.92, 9.91, 9.94, 9.93, 9.94, 9.93, 9.94]
k = np.arange(1, 10)
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.axhline(10, color=INK2, lw=1, ls=(0, (4, 3))); ax.annotate("maximum: 10 bits", (9, 10), xytext=(0, 5), textcoords="offset points", ha="right", color=INK2, fontsize=10)
ax.plot(k, dac, "-o", color=BLUE, ms=8, mec=SURF, mew=2, label="DAC")
ax.plot(k, enc, "-o", color=ORANGE, ms=8, mec=SURF, mew=2, label="EnCodec")
ax.annotate("DAC", (9, dac[-1]), xytext=(8, -4), textcoords="offset points", color=INK); ax.annotate("EnCodec", (9, enc[-1]), xytext=(8, -4), textcoords="offset points", color=INK)
ax.set_xticks(k); ax.set_xlim(0.6, 10.2); ax.set_ylim(8.2, 10.2); ax.set_xlabel("Codebook"); ax.set_ylabel("Information per index (bits)")
fig.tight_layout(); fig.savefig("out/fig_codebook_usage.png", dpi=200); plt.close(fig)
