"""Concept figure: the order in which each family brings the music into existence."""
import os, sys; sys.path.insert(0, ".."); os.makedirs("out", exist_ok=True)
import numpy as np
from matplotlib.patches import Rectangle
from style import *

rng = np.random.default_rng(0); N = 12
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
def grid(ax, title, fill):
    for r in range(4):
        for c in range(N):
            done, noise = fill(r, c)
            col = BLUE if done else SURF
            ax.add_patch(Rectangle((c, 3 - r), 0.9, 0.8, facecolor=col, edgecolor=GRID if not done else BLUE, lw=1, alpha=1 if done else 1))
            if noise > 0:
                ax.add_patch(Rectangle((c, 3 - r), 0.9, 0.8, facecolor=BLUE, edgecolor="none", alpha=1 - noise))
                pts = rng.random((int(14 * noise), 2)); ax.scatter(c + 0.05 + pts[:, 0] * 0.8, 3 - r + 0.05 + pts[:, 1] * 0.7, s=2, color=INK2)
    for r, lab in enumerate(["start", "", "", "end"]): ax.text(-0.3, 3 - r + 0.4, lab, ha="right", va="center", color=INK2, fontsize=10)
    ax.annotate("", xy=(N, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="->", color=INK2, lw=1)); ax.text(N / 2, -0.95, "time in the music", ha="center", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", fontsize=11); ax.set_xlim(-1.6, N + 0.2); ax.set_ylim(-1.2, 4.1); ax.axis("off")
grid(axs[0], "Autoregressive: left to right", lambda r, c: (c < [1, 4, 8, 12][r], 0))
grid(axs[1], "Diffusion and flow matching: everything at once", lambda r, c: (False, [1.0, 0.66, 0.33, 0.0][r]) if r < 3 else (True, 0))
fig.text(0.5, 0.01, "Each row is a later moment during generation. Blue = finished, speckled = still noisy.", ha="center", color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("out/fig_order.png", dpi=200); plt.close(fig)
