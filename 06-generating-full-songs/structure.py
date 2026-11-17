"""Does it come back? Self-similarity of real songs and of a stretched MusicGen piece.

A self-similarity matrix compares every moment of a recording with every other moment.
A chorus that returns shows up as a stripe parallel to the main diagonal.

Second score, 'last vs first': for every beat in the last third (30 s of a 90 s piece),
the best match in the first third, averaged. A chained MusicGen piece never sees its own first
30 seconds again after second 50, so this is where forgetting should show.

The 'return score' of a recording: for every beat, how similar is the best-matching
passage at least 15 seconds away? Averaged over beats. A song whose sections recur
scores high; music that keeps moving on to new material scores low.

Run after generate_long.py:  python structure.py
"""
import json, os, sys
import numpy as np, librosa, scipy.signal
if not hasattr(scipy.signal, "hann"): scipy.signal.hann = scipy.signal.windows.hann   # old librosa + new scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
AUDIO = os.path.join(HERE, "..", "data", "audio")
MIN_LAG = 15.0                                              # seconds

def features(y, sr):
    """Beat-synchronous chroma (which pitch classes sound) plus MFCC (timbre),
    stacked over 4 beats so that a match means a matching passage, not one chord."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)[1:]
    f = np.vstack([chroma / (np.linalg.norm(chroma, axis=0) + 1e-9),
                   mfcc / (np.linalg.norm(mfcc, axis=0) + 1e-9)])
    f = librosa.util.sync(f, beats, aggregate=np.median)
    f = f - f.mean(axis=1, keepdims=True)                   # centre, so unrelated passages score near 0
    f = librosa.feature.stack_memory(f, n_steps=4, mode="edge")
    f = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-9)
    times = np.concatenate([[0.0], librosa.frames_to_time(beats, sr=sr, hop_length=512)])   # segment start times
    return f, times[: f.shape[1]], float(np.atleast_1d(tempo)[0])

def analyse(path, seconds=90):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=seconds)
    f, t, tempo = features(y, sr)
    S = f.T @ f                                              # cosine similarity, beats x beats
    far = np.abs(t[:, None] - t[None, :]) >= MIN_LAG
    score = float(np.mean(np.where(far, S, -1).max(axis=1)))
    dur = len(y) / sr
    first, last = t < dur / 3, t >= 2 * dur / 3               # does the last third revisit the first third?
    back = float(np.mean(S[np.ix_(last, first)].max(axis=1)))
    return S, t, tempo, score, back

recs = {"Real song with vocals (fishin)": os.path.join(AUDIO, "fishin.wav"),
        "Real instrumental (sweetwaltz)": os.path.join(AUDIO, "sweetwaltz.wav"),
        "Real instrumental (pistachio)": os.path.join(AUDIO, "pistachio.wav"),
        **{f"MusicGen chained to 90 s, seed {k}": os.path.join(HERE, "out", f"musicgen_90s_s{k}.wav") for k in range(3)}}
res = {}
for name, p in recs.items():
    if not os.path.exists(p): print("missing", p, "- run generate_long.py first"); continue
    S, t, tempo, score, back = analyse(p)
    res[name] = {"score": round(score, 3), "last_vs_first": round(back, 3), "tempo": round(tempo, 1), "beats": len(t), "seconds": round(float(t[-1]), 1)}
    np.save(os.path.join(HERE, "out", f"ssm_{len(res)}.npy"), np.vstack([t[None], S]))   # numbered in the order above
    print(f"{name:45s} return score {score:.3f}   last third vs first third {back:.3f}   ({len(t)} beats, {tempo:.0f} BPM)")
mg = [v for k, v in res.items() if k.startswith("MusicGen")]
if mg: print(f"{'MusicGen mean of ' + str(len(mg)) + ' seeds':45s} return score {np.mean([v['score'] for v in mg]):.3f}   last third vs first third {np.mean([v['last_vs_first'] for v in mg]):.3f}")
json.dump(res, open(os.path.join(HERE, "structure.json"), "w"), indent=1)
