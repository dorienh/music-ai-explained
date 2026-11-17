"""Draws the article's figures from the saved results. Run after scaling.py and train_compare.py."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import plt, BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF
IMG = os.path.join(HERE, "..", "..", "images", "09")
if not os.path.isdir(os.path.join(HERE, "..", "..", "images")): IMG = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
save = lambda fig, name: (fig.savefig(os.path.join(IMG, name), dpi=200, bbox_inches="tight"), print("wrote", name))

# ---------- 1. time and memory against length ----------
res = json.load(open(os.path.join(HERE, "scaling.json")))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, ylabel in ((axes[0], "seconds", "Time for one forward pass (s)"), (axes[1], "extra_mb", "Extra memory (MB)")):
    for kind, col, label in (("attention", ORANGE, "Attention"), ("state space", BLUE, "State space")):
        pts = [(r["n"], max(r[key], 1e-3)) for r in res if r["kind"] == kind]
        ax.plot(*zip(*pts), color=col, marker="o", label=label)
        ax.annotate(label, pts[-1], textcoords="offset points", xytext=(6, 0), va="center", color=col, fontsize=10)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylabel(ylabel); ax.set_xlabel("Sequence length (frames)")
    ticks = [512, 2048, 8192, 32768]; ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks]); ax.set_xlim(400, 220000)
axes[0].set_title("Attention is faster at every length I could run", loc="left", fontsize=10)
axes[1].set_title("and needs ten times the memory at 32,768 frames", loc="left", fontsize=10)
fig.tight_layout(); save(fig, "fig_scaling.png")

# ---------- 2. two backbones on the same music ----------
p = os.path.join(HERE, "train_compare.json")
if os.path.exists(p):
    r = json.load(open(p))
    if "State space" in r:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for name, col in (("Transformer", ORANGE), ("State space", BLUE)):
            c = r[name]["curve"]
            axes[0].plot([x[0] for x in c], [x[2] for x in c], color=col, marker="o", ms=4, label=name)
            axes[1].plot([x[1] / 60 for x in c], [x[2] for x in c], color=col, marker="o", ms=4, label=name)
        axes[0].set_xlabel("Training steps"); axes[1].set_xlabel("Minutes of training on a laptop CPU"); axes[0].set_ylabel("Held-out loss (nats per token)")
        axes[0].set_title("Per step, the state space model learns faster at first, then overfits", loc="left", fontsize=10); axes[1].set_title("Per minute on this laptop, the transformer is far ahead", loc="left", fontsize=10)
        axes[0].legend(frameon=False); fig.tight_layout(); save(fig, "fig_training.png")
