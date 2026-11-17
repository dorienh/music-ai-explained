"""Shared helpers for this folder: the Bach chorales as plain note lists, a MIDI writer and a tiny synthesizer.

A note is (start, duration, pitch): start and duration in sixteenth notes, pitch as a MIDI number (60 = middle C).
The chorales come from the music21 corpus (371 four-part chorales by J. S. Bach, public domain).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
BPM = 80                                   # tempo used whenever a chorale is turned into seconds
SIXTEENTH = 60 / BPM / 4                   # seconds per sixteenth note

def load_chorales():
    """Returns a list of dicts: title, tonic (pitch class 0-11), mode, notes [(start, dur, pitch), ...]. Cached after the first call."""
    cache = os.path.join(OUT, "chorales.json")
    if os.path.exists(cache): return json.load(open(cache))
    import music21 as m21
    res = []
    for c in m21.corpus.chorales.Iterator():
        key = c.analyze("key"); notes = []
        for n in c.flatten().notes:
            start, dur = float(n.offset) * 4, float(n.duration.quarterLength) * 4
            if dur <= 0: continue
            for p in n.pitches: notes.append((round(start), max(1, round(dur)), p.midi))
        res.append({"title": c.metadata.title or "", "tonic": key.tonic.pitchClass, "mode": key.mode, "notes": sorted(set(notes))})
    json.dump(res, open(cache, "w"))
    return res

def write_midi(notes, path, bpm=BPM, velocity=80):
    """Notes in sixteenths -> a standard MIDI file with one piano track."""
    import mido
    tpq = 480; step = tpq // 4
    ev = sorted([(s * step, 1, p) for s, d, p in notes] + [((s + d) * step, 0, p) for s, d, p in notes])   # note-offs before note-ons at the same tick
    mid = mido.MidiFile(ticks_per_beat=tpq); tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    t = 0
    for tick, on, p in ev:
        tr.append(mido.Message("note_on" if on else "note_off", note=int(p), velocity=velocity if on else 0, time=tick - t)); t = tick
    mid.save(path)
    return os.path.getsize(path)

def synth(notes_sec, sr=22050, harmonics=(1, 0.5, 0.33, 0.25, 0.2), decay=1.5):
    """Notes as (start_seconds, duration_seconds, midi_pitch) -> a waveform.
    Each note is a sum of a few sine waves (the harmonics) that fades out; change `harmonics` to change the instrument."""
    end = max(s + d for s, d, _ in notes_sec) + 0.3
    y = np.zeros(int(end * sr))
    for s, d, p in notes_sec:
        f = 440 * 2 ** ((p - 69) / 12); n = int(d * sr); t = np.arange(n) / sr
        tone = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics) if f * (k + 1) < sr / 2)
        env = np.exp(-decay * t) * np.minimum(1, t / 0.01) * np.minimum(1, (d - t) / 0.02)   # 10 ms attack, 20 ms release
        i = int(s * sr); y[i:i + n] += tone * env
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)

def to_seconds(notes):
    return [(s * SIXTEENTH, d * SIXTEENTH, p) for s, d, p in notes]
