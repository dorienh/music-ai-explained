"""Why song models tokenize vocals and accompaniment separately.

Speech is mixed with music at a range of vocal-to-accompaniment ratios (VAR), then
passed through MusicGen's codec (EnCodec 32 kHz, 4 codebooks, 2.2 kbit/s) in two ways:

  mixed   - the whole mix becomes one token stream (as in Jukebox or MusicGen)
  split   - vocal and accompaniment are coded separately and summed after decoding
            (the idea behind YuE's track-decoupled tokens)

Word error rate (WER) of a speech recogniser measures how many words survive.
The reference transcript is the recogniser's output on the clean speech.

Run:  python vocal_masking.py          (about 10 minutes on a laptop CPU)
Needs: transformers, faster-whisper, librosa, soundfile
"""
import json, os, re, sys
import numpy as np, librosa, torch
from transformers import EncodecModel
from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "..", "data", "audio")          # filled by data/get_audio.py
SR = 32000
VARS = [None, 6, 0, -4, -8, -12]                           # None = vocal alone

def load(name, seconds=None):
    y, _ = librosa.load(os.path.join(AUDIO, name + ".wav"), sr=SR, mono=True)
    return y if seconds is None else y[: int(seconds * SR)]

def rms(x): return np.sqrt(np.mean(x ** 2) + 1e-12)

codec = EncodecModel.from_pretrained("facebook/encodec_32khz").eval()
def through_codec(x):
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)[None, None]
        enc = codec.encode(t)
        y = codec.decode(enc.audio_codes, enc.audio_scales)[0][0, 0].numpy()
    return y[: len(x)]

asr = WhisperModel("medium", device="cpu", compute_type="int8")
def transcribe(x):
    x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
    segs, _ = asr.transcribe(x16.astype(np.float32), language="en", beam_size=5)
    return " ".join(s.text for s in segs)

def words(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):                          # Levenshtein distance on words
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)

voices = {"libri1": load("libri1"), "libri2": load("libri2")}
music = {"choice": load("choice"), "vibeace": load("vibeace", 20)}
refs = {k: transcribe(v) for k, v in voices.items()}
for k, v in refs.items(): print(f"reference {k}: {v[:90]}...")

rows = []
for vname, v in voices.items():
    for mname, m in music.items():
        m = np.resize(m, len(v))                             # loop or cut music to the speech length
        for var in VARS:
            a = np.zeros_like(v) if var is None else m * rms(v) / rms(m) / 10 ** (var / 20)
            peak = max(np.abs(v + a).max(), 1e-9) / 0.9      # same gain for every version
            v_, a_ = v / peak, a / peak
            mix = v_ + a_
            out = {"voice": vname, "music": mname, "var": var,
                   "raw": wer(refs[vname], transcribe(mix)),
                   "mixed": wer(refs[vname], transcribe(through_codec(mix))),
                   "split": wer(refs[vname], transcribe(through_codec(v_) + (through_codec(a_) if var is not None else 0)))}
            rows.append(out); print(out, flush=True)

json.dump(rows, open(os.path.join(HERE, "vocal_masking.json"), "w"), indent=1)
print("\nmean WER (%) over 4 voice/music pairs")
print(f"{'VAR':>8} {'raw mix':>8} {'mixed tokens':>13} {'split tokens':>13}")
for var in VARS:
    sel = [r for r in rows if r["var"] == var]
    print(f"{'alone' if var is None else f'{var:+d} dB':>8}", *(f"{100*np.mean([r[k] for r in sel]):>{w}.1f}" for k, w in (("raw", 8), ("mixed", 13), ("split", 13))))
