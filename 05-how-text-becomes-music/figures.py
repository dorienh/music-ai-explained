"""Figures for 'How Text Becomes Music'. Reads clap_matrix.json (experiments.py) and guidance_vs_prompt.json."""
import os, sys, json; sys.path.insert(0, ".."); os.makedirs("out", exist_ok=True)
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style import *

d = json.load(open("clap_matrix.json")); M = np.array(d["sim"])
labels = ["solo trumpet", "string orchestra", "celesta", "drum and bass", "folk pop song", "ragtime piano", "jazz combo", "speech"]
cmap = LinearSegmentedColormap.from_list("blue", [SURF, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.imshow(M, cmap=cmap, vmin=-0.1, vmax=0.65)
for i in range(len(M)):
    for j in range(len(M)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9.5, color="white" if M[i, j] > 0.42 else INK,
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(8)); ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticks(range(8)); ax.set_yticklabels([f"“{c}”" for c in d["captions"]])
ax.set_xlabel("Audio clip"); ax.set_title("Caption", loc="left", x=-0.42, fontsize=11, color=INK2); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, 8), minor=True); ax.set_yticks(np.arange(-.5, 8), minor=True); ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="both", length=0)
fig.tight_layout(); fig.savefig("out/fig_clap_matrix.png", dpi=200); plt.close(fig)

if os.path.exists("guidance_vs_prompt.json"):
    rows = json.load(open("guidance_vs_prompt.json")); scales = sorted({r["scale"] for r in rows})
    own = [np.mean([r["own"] for r in rows if r["scale"] == s]) for s in scales]; oth = [np.mean([r["others"] for r in rows if r["scale"] == s]) for s in scales]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(scales, own, "-o", color=BLUE, ms=8, mec=SURF, mew=2, label="Similarity to its own prompt")
    ax.plot(scales, oth, "-o", color=ORANGE, ms=8, mec=SURF, mew=2, label="Similarity to the other prompts")
    for s in sorted({r["prompt"] for r in rows}):
        ax.plot(scales, [np.mean([r["own"] for r in rows if r["scale"] == k and r["prompt"] == s]) for k in scales], color=BLUE, lw=1, alpha=0.3)
    ax.set_xticks(scales); ax.set_xticklabels([f"{s:g}" for s in scales]); ax.set_ylim(-0.05, 0.7)
    ax.set_xlabel("Guidance scale"); ax.set_ylabel("CLAP similarity"); ax.legend(frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig("out/fig_guidance.png", dpi=200); plt.close(fig)
