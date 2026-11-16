"""The two calculations from the article that need no model.

1. How long is a song, in the units each model works with?
2. How lyrics are laid out against time: sentence starts (DiffRhythm) versus
   word starts and ends (JAM), on the 21.5 Hz latent grid both models use.

Run:  python by_hand.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. sequence lengths for a 3-minute song ----------
SONG = 180  # seconds
systems = [  # name, frames per second, symbols per frame, what one symbol is
    ("Raw audio, 44.1 kHz stereo", 44100, 2, "16-bit samples"),
    ("MusicGen (EnCodec 32 kHz, 4 codebooks)", 50, 4, "codebook indices"),
    ("YuE stage 1 (X-Codec codebook 0, vocal + accompaniment)", 50, 2, "codebook indices"),
    ("YuE stage 2 (X-Codec codebooks 0-7, two tracks)", 50, 16, "codebook indices"),
    ("DiffRhythm / JAM (VAE latent, 64 numbers per frame)", 44100 / 2048, 1, "64-number vectors"),
]
print(f"A {SONG}-second song is:")
for name, fps, per, unit in systems:
    print(f"  {name:58s} {fps*SONG:>9,.0f} frames x {per:>2} = {fps*SONG*per:>11,.0f} {unit}")
print(f"  MusicGen's 30-second context covers {30/SONG:.0%} of the song")

# ---------- 2. lyrics on the latent grid ----------
FPS = 44100 / 2048                       # 21.53 latent frames per second
LEX = {"hold": "HH OW L D", "on": "AA N", "to": "T UW", "me": "M IY",
       "the": "DH AH", "night": "N AY T", "is": "IH Z", "long": "L AO NG"}
# word, start (s), end (s): how a singer might phrase it, with "me" and "long" held
WORDS = [("hold", 2.00, 2.40), ("on", 2.40, 2.70), ("to", 2.70, 2.90), ("me", 2.90, 3.80),
         ("the", 5.00, 5.15), ("night", 5.15, 5.70), ("is", 5.70, 5.90), ("long", 5.90, 7.00)]
LINES = [(2.00, ["hold", "on", "to", "me"]), (5.00, ["the", "night", "is", "long"])]
N = int(7.5 * FPS)
frame = lambda t: int(t * FPS)

# DiffRhythm: write each line's phonemes one per frame, starting at the line's start frame
dr = ["."] * N
for start, ws in LINES:
    ph = [p for w in ws for p in LEX[w].split()]
    f0 = frame(start)
    dr[f0: f0 + len(ph)] = ph

# JAM: spread each word's phonemes evenly over that word's frames, "-" = still inside the word
jam = ["."] * N
for w, s, e in WORDS:
    ph, f0, f1 = LEX[w].split(), frame(s), frame(e)
    for k in range(f0, f1): jam[k] = "-"
    for i, p in enumerate(ph): jam[f0 + int(i * (f1 - f0) / len(ph))] = p

def show(name, seq):
    print(f"\n{name}")
    for a in range(frame(1.8), frame(7.2), 20):
        print(f"  {a/FPS:4.2f}s  " + " ".join(f"{x:>2}" for x in seq[a:a + 20]))
print(f"\nlatent frame = {1000/FPS:.1f} ms; '.' = no vocal")
show("Sentence-level (DiffRhythm): phonemes packed at each line's start", dr)
show("Word-level (JAM): phonemes spread over each word's span", jam)
sung = sum((e - s) for _, s, e in WORDS)
print(f"\nline 1 has {sum(len(LEX[w].split()) for w in LINES[0][1])} phonemes; packed they fill "
      f"{sum(len(LEX[w].split()) for w in LINES[0][1])/FPS:.2f} s, the singer takes {3.80-2.00:.2f} s")
json.dump({"fps": FPS, "dr": dr, "jam": jam, "words": WORDS}, open(os.path.join(HERE, "by_hand.json"), "w"))
