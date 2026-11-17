"""What do notes keep, and what do they throw away? A trumpet recording turned into notes and back.

1. Track the pitch of a solo trumpet recording and cut it into notes (a simple transcription).
2. Save those notes as MIDI and play them back with two different synthetic instruments.
3. Measure what survived: the pitch content (chroma) and what did not: the sound colour (MFCC, CLAP).

Run:  python round_trip.py      (needs the test clips: python ../data/get_audio.py; about a minute on a CPU)
"""
import json, os, sys
import numpy as np, librosa, soundfile as sf, scipy.signal
if not hasattr(scipy.signal, "hann"): scipy.signal.hann = scipy.signal.windows.hann
from common import write_midi, synth, OUT, HERE

SR, HOP = 22050, 256
y, _ = librosa.load(os.path.join(HERE, "..", "data", "audio", "trumpet.wav"), sr=SR, mono=True)
dur = len(y) / SR

# ---------- 1. audio -> notes ----------
f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C3"), fmax=librosa.note_to_hz("C7"), sr=SR, hop_length=HOP)
midi = librosa.hz_to_midi(f0)                                # fractional pitch per frame, NaN where no pitch
frame = HOP / SR
notes, start = [], None
for i in range(len(midi) + 1):
    cur = None if i == len(midi) or np.isnan(midi[i]) else int(np.round(midi[i]))
    if start is not None and cur != pitch:                   # the current note ends here
        if (i - start) * frame >= 0.05: notes.append((start * frame, (i - start) * frame, pitch, start, i))
        start = None
    if cur is not None and start is None: start, pitch = i, cur
print(f"== 1. {dur:.2f} s of trumpet -> {len(notes)} notes")
print("   ", " ".join(librosa.midi_to_note(p) for _, _, p, _, _ in notes))

# what one number per note leaves out: how far the real pitch strays from the note it was rounded to
cents = np.concatenate([100 * (midi[a:b] - p) for _, _, p, a, b in notes])
rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
print(f"    inside the notes, the played pitch is on average {np.mean(np.abs(cents)):.0f} cents from the written one (95% within {np.percentile(np.abs(cents), 95):.0f} cents)")
swing = float(np.mean([np.ptp(20 * np.log10(rms[a + 2:b - 2] + 1e-9)) for _, _, _, a, b in notes if b - a > 8]))   # skip each note's first and last 23 ms
print(f"    and the loudness inside one note moves by {swing:.0f} dB on average; the note list stores one pitch and one loudness per note")

# ---------- 2. notes -> MIDI -> audio, twice ----------
tick = 0.01                                                   # write the MIDI on a 10 ms grid
grid = [(round(s / tick), max(1, round(d / tick)), p) for s, d, p, _, _ in notes]
mbytes = write_midi(grid, os.path.join(OUT, "trumpet.mid"), bpm=60 / (tick * 4))      # one sixteenth = 10 ms
wbytes = len(y) * 2                                            # 16-bit mono at 22.05 kHz
print(f"\n== 2. Size: recording {wbytes:,} bytes (16-bit mono, 22.05 kHz), notes as MIDI {mbytes:,} bytes, {wbytes / mbytes:.0f} x smaller")
nsec = [(s, d, p) for s, d, p, _, _ in notes]
organ = synth(nsec, SR, harmonics=(1, 0.5, 0.33, 0.25, 0.2), decay=0.5)
bell = synth(nsec, SR, harmonics=(1, 0, 0.6, 0, 0.4, 0, 0.3), decay=6)
for name, w in (("trumpet_original", y), ("trumpet_notes_as_organ", organ), ("trumpet_notes_as_bell", bell)):
    sf.write(os.path.join(OUT, name + ".wav"), w, SR)

# ---------- 3. what survived ----------
def chroma(w): return librosa.feature.chroma_cqt(y=w, sr=SR, hop_length=512)
def mfcc(w): return librosa.feature.mfcc(y=w, sr=SR, n_mfcc=13, hop_length=512)[1:]
def framewise(a, b, loud_only=None):
    n = min(a.shape[1], b.shape[1]); a, b = a[:, :n], b[:, :n]
    c = np.sum(a * b, 0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
    return float(np.mean(c[loud_only[:n]])) if loud_only is not None else float(np.mean(c))
on = librosa.feature.rms(y=y, hop_length=512)[0] > 0.02      # compare only where the trumpet is playing
rng = np.random.default_rng(0)
shuffled = synth([(s, d, p + int(rng.integers(1, 11))) for s, d, p in nsec], SR)     # same rhythm, wrong notes
res = {}
print("\n== 3. Similarity to the original recording, frame by frame (1 = identical)")
print(f"    {'':34s} {'pitch content':>14s} {'sound colour':>13s}")
print(f"    {'':34s} {'(chroma)':>14s} {'(MFCC)':>13s}")
for name, w in (("the same notes on an organ-like synth", organ), ("the same notes on a bell-like synth", bell), ("same rhythm, wrong notes", shuffled)):
    c, m = framewise(chroma(y), chroma(w), on), framewise(mfcc(y), mfcc(w), on)
    res[name] = {"chroma": c, "mfcc": m}; print(f"    {name:38s} {c:>10.2f} {m:>13.2f}")

try:                                                           # a second opinion from CLAP: which instrument is this?
    import torch
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval(); proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    labels = ["trumpet", "organ", "bell", "piano", "violin"]
    with torch.no_grad():
        T = torch.nn.functional.normalize(clap.get_text_features(**proc(text=[f"a solo {l}" for l in labels], return_tensors="pt", padding=True)), dim=-1)
        print("\n    CLAP similarity between the audio and five instrument labels:")
        for name, w in (("original", y), ("organ-like synth", organ), ("bell-like synth", bell)):
            a = torch.nn.functional.normalize(clap.get_audio_features(**proc(audios=[librosa.resample(w, orig_sr=SR, target_sr=48000)], sampling_rate=48000, return_tensors="pt")), dim=-1)
            cos = (a @ T.T)[0].numpy()
            res["clap_" + name] = dict(zip(labels, map(float, cos)))
            print(f"    {name:18s}", "  ".join(f"{l} {v:.2f}" for l, v in zip(labels, cos)), "  -> best match:", labels[int(cos.argmax())])
except Exception as e: print("    (CLAP skipped:", repr(e)[:80], ")")

json.dump({"seconds": dur, "notes": [(round(s, 3), round(d, 3), int(p)) for s, d, p in nsec], "wav_bytes": wbytes, "midi_bytes": mbytes,
           "cents_mean_abs": float(np.mean(np.abs(cents))), "cents_p95": float(np.percentile(np.abs(cents), 95)),
           "loudness_swing_db": swing, "similarity": res},
          open(os.path.join(HERE, "round_trip.json"), "w"), indent=1)
np.save(os.path.join(OUT, "trumpet_f0.npy"), midi)
