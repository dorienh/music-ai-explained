"""Draws the article's figures from the saved results. Run after measure_local.py and precision.py."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import plt, BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF
IMG = os.path.join(HERE, "..", "..", "images", "08")
if not os.path.isdir(os.path.join(HERE, "..", "..", "images")): IMG = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
save = lambda fig, name: (fig.savefig(os.path.join(IMG, name), dpi=200, bbox_inches="tight"), print("wrote", name))

# ---------- 1. memory: predicted from the parameter count against measured ----------
m = json.load(open(os.path.join(HERE, "measure_local.json")))
rows = [(k.split("/")[-1].replace(" | cpu", "").replace("musicgen", "MusicGen"), v) for k, v in m.items() if v["device"] == "cpu"]   # the Apple GPU reading is not comparable
fig, ax = plt.subplots(figsize=(9.5, 1.3 + 0.95 * len(rows)))
for i, (name, v) in enumerate(rows):
    used = v["peak_ram_gb"]
    ax.barh(i - 0.19, v["weights_gb_fp32"], height=0.36, color=BLUE); ax.barh(i + 0.19, used, height=0.36, color=ORANGE)
    ax.text(v["weights_gb_fp32"] + 0.08, i - 0.19, f"{v['weights_gb_fp32']:.1f} GB", va="center", fontsize=10)
    ax.text(used + 0.08, i + 0.19, f"{used:.1f} GB", va="center", fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(v["peak_ram_gb"] for _, v in rows) * 1.2); ax.set_xlabel("Memory (GB)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["Predicted: parameters x 4 bytes", "Measured peak while generating 10 s"], frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); save(fig, "fig_memory.png")

# ---------- 2. bits per weight against quality ----------
p = json.load(open(os.path.join(HERE, "precision.json")))["results"]
bits = sorted(map(int, p), reverse=True); loss = [p[str(b)]["loss"] for b in bits]; agree = [100 * p[str(b)]["agree"] for b in bits]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
x = range(len(bits))
axes[0].plot(x, loss, color=BLUE, marker="o"); axes[0].axhline(np.log(2048), color=INK2, lw=1, ls="--")
axes[0].text(0.05, np.log(2048) - 0.12, "uniform guess over 2,048 tokens", color=INK2, fontsize=9, va="top")
axes[0].set_ylabel("Loss on real music (nats per token)"); axes[0].set_title("Lower is better", loc="left", fontsize=10)
axes[1].plot(x, agree, color=ORANGE, marker="o"); axes[1].set_ylim(0, 115)
axes[1].set_ylabel("Same top guess as the 32-bit model (%)"); axes[1].set_title("Higher is better", loc="left", fontsize=10)
for ax, vals, fmt in ((axes[0], loss, "{:.2f}"), (axes[1], agree, "{:.0f}%")):
    ax.set_xticks(list(x)); ax.set_xticklabels([str(b) for b in bits]); ax.set_xlabel("Bits per weight")
    for i, v in enumerate(vals): ax.annotate(fmt.format(v), (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
axes[0].set_ylim(min(loss) - 0.4, max(max(loss), np.log(2048)) + 0.5)
fig.tight_layout(); save(fig, "fig_precision.png")
