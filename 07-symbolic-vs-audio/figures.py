"""Draws the article's figures from the saved results.
Run after round_trip.py, tokenizations.py and tiny_model.py.
"""
import json, os, sys
import numpy as np, librosa, scipy.signal
if not hasattr(scipy.signal, "hann"): scipy.signal.hann = scipy.signal.windows.hann
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from style import plt, BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF
IMG = os.path.join(HERE, "..", "..", "images", "07")
if not os.path.isdir(os.path.join(HERE, "..", "..", "images")): IMG = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
save = lambda fig, name: (fig.savefig(os.path.join(IMG, name), dpi=200, bbox_inches="tight"), print("wrote", name))

# ---------- 1. a recording, its pitch, and its notes ----------
SR, HOP = 22050, 256
y, _ = librosa.load(os.path.join(HERE, "..", "data", "audio", "trumpet.wav"), sr=SR, mono=True)
rt = json.load(open(os.path.join(HERE, "round_trip.json"))); f0 = np.load(os.path.join(HERE, "out", "trumpet_f0.npy"))
C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C3"), n_bins=60 * 3, bins_per_octave=36))
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
ax = axes[0]
ax.imshow(librosa.amplitude_to_db(C, ref=np.max), origin="lower", aspect="auto", cmap="Greys", vmin=-55, vmax=0,
          extent=[0, len(y) / SR, 48, 48 + 60])
ax.set_ylim(60, 108); ax.grid(False); ax.set_ylabel("Pitch (MIDI number)")
ax.set_title("The recording: every stripe is energy at some frequency. The lowest stripe of each note is its pitch, the ones above are overtones", loc="left", fontsize=10)
ax = axes[1]
t = np.arange(len(f0)) * HOP / SR
ax.plot(t, f0, color=ORANGE, lw=1.6, label="Pitch as played (tracked from the recording)")
for i, (s, d, p) in enumerate(rt["notes"]):
    ax.add_patch(plt.Rectangle((s, p - 0.5), d, 1, color=BLUE, alpha=0.35, lw=0, label="Notes: one pitch, one start, one length" if i == 0 else None))
ax.set_ylim(63, 77); ax.set_yticks([65, 68, 70, 72, 75]); ax.set_yticklabels(["F4", "G♯4", "A♯4", "C5", "D♯5"])
ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch"); ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title(f"The notes: {len(rt['notes'])} boxes, {rt['midi_bytes']} bytes as MIDI against {rt['wav_bytes']:,} bytes of audio", loc="left", fontsize=10)
fig.tight_layout(); save(fig, "fig_notes_vs_sound.png")

# ---------- 2. tokens per second of music ----------
tk = json.load(open(os.path.join(HERE, "tokenizations.json")))
rows = [("REMI + BPE (5,000)", tk["REMI + BPE (5000)"]["per_second"], BLUE), ("Octuple", tk["Octuple"]["per_second"], BLUE),
        ("Compound Word", tk["Compound Word"]["per_second"], BLUE), ("MIDI-like", tk["MIDI-like"]["per_second"], BLUE), ("REMI", tk["REMI"]["per_second"], BLUE),
        ("Stable Audio latent frames", 21.5, ORANGE), ("EnCodec 32 kHz, frames", 50, ORANGE), ("EnCodec 32 kHz, all 4 codebooks", 200, ORANGE), ("DAC 44.1 kHz, all 9 codebooks", 774, ORANGE)]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
for i, (n, v, c) in enumerate(rows):
    ax.barh(i, v, color=c, height=0.62); ax.text(v * 1.08, i, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(1, 2500)
ax.set_xlabel("Model steps per second of music (log scale)"); ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["Notes (Bach chorales, 5.9 notes per second)", "Sound (any music, fixed rate)"], frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); save(fig, "fig_tokens_per_second.png")

# ---------- 3. piano rolls: real, transformer, bigram ----------
fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, sharey=True)
for ax, name, title, col in ((axes[0], "chorale_real", "A real chorale the model never saw", INK2),
                             (axes[1], "chorale_generated", "Tiny transformer (850,000 parameters, trained on a laptop)", BLUE),
                             (axes[2], "chorale_bigram", "Bigram table: each token chosen from the one before it", ORANGE)):
    for s, d, p in json.load(open(os.path.join(HERE, "out", name + ".json"))):
        if s < 16 * 8: ax.add_patch(plt.Rectangle((s / 16, p - 0.45), d / 16 * 0.96, 0.9, color=col, lw=0, zorder=3))
    ax.set_axisbelow(True); ax.set_xlim(0, 8); ax.set_ylim(34, 86); ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel("Pitch")
    ax.set_xticks(range(9)); ax.set_yticks([36, 48, 60, 72, 84]); ax.set_yticklabels(["C2", "C3", "C4", "C5", "C6"])
axes[2].set_xlabel("Bars")
fig.tight_layout(); save(fig, "fig_chorale_rolls.png")
