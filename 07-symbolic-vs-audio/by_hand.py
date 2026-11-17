"""The same four notes written six ways, and one Bach chorale as notes and as sound.

Run:  python by_hand.py     (a few seconds; the first run reads the chorales from music21, about 20 s)
"""
import json, os
from common import load_chorales, write_midi, OUT, HERE, BPM, SIXTEENTH

# ---------- 1. four quarter notes: C4 E4 G4 C5, one bar of 4/4 at 120 BPM ----------
phrase = [(0, 4, 60), (4, 4, 64), (8, 4, 67), (12, 4, 72)]          # (start, duration, pitch) in sixteenths
bpm = 120; seconds = 16 * 60 / bpm / 4
print(f"== 1. One bar, four notes, {seconds:.0f} seconds at {bpm} BPM")

# MIDI-like events (Performance RNN / Music Transformer): note-on, note-off, time-shift, velocity
midi_like = ["VELOCITY_80"]
for s, d, p in phrase: midi_like += [f"NOTE_ON_{p}", f"TIME_SHIFT_{int(d * 60 / bpm / 4 * 1000)}ms", f"NOTE_OFF_{p}"]
# REMI (Pop Music Transformer): bar, position in the bar, then pitch, velocity, duration for each note
remi = ["BAR"]
for s, d, p in phrase: remi += [f"POSITION_{s + 1}/16", f"PITCH_{p}", "VELOCITY_80", f"DURATION_{d}/16"]
# compound word / Octuple: one token per note, made of several fields
compound = [f"(bar 1, position {s + 1}/16, pitch {p}, duration {d}/16, velocity 80)" for s, d, p in phrase]
abc = "C E G c |"                                                   # with the header L:1/4, K:C
roll_cells = 16 * 88
midi_bytes = write_midi(phrase, os.path.join(OUT, "phrase.mid"), bpm=bpm)

rows = [("MIDI file", midi_bytes, "bytes"), ("MIDI-like events", len(midi_like), "tokens"), ("REMI", len(remi), "tokens"),
        ("Compound word / Octuple", len(compound), "tokens of 5 fields"), ("ABC notation", len(abc.replace(" ", "")), "characters"),
        ("Piano roll, 16 steps x 88 keys", roll_cells, "cells, 16 of them on"),
        ("EnCodec tokens (50 frames/s x 4 codebooks)", int(seconds * 50 * 4), "tokens"),
        ("CD audio (44,100 samples/s x 2 channels)", int(seconds * 44100 * 2), "samples")]
for n, v, u in rows: print(f"  {n:45s} {v:>9,} {u}")
print("  MIDI-like:", " ".join(midi_like))
print("  REMI:     ", " ".join(remi))
print("  ABC:      ", abc)

# ---------- 2. a whole chorale ----------
ch = load_chorales()
c = ch[0]
notes = c["notes"]; length16 = max(s + d for s, d, _ in notes); sec = length16 * SIXTEENTH
mb = write_midi(notes, os.path.join(OUT, "chorale.mid"))
wav = int(sec * 44100 * 2 * 2); mp3 = int(sec * 128000 / 8)
print(f"\n== 2. One chorale: '{c['title']}', {len(notes)} notes, {length16 // 4} beats, {sec:.0f} s at {BPM} BPM")
print(f"  MIDI file            {mb:>10,} bytes  ({mb / len(notes):.1f} bytes per note, {mb * 8 / sec:.0f} bit/s)")
print(f"  MP3 at 128 kbit/s    {mp3:>10,} bytes  ({mp3 / mb:.0f} x the MIDI file)")
print(f"  CD-quality WAV       {wav:>10,} bytes  ({wav / mb:.0f} x the MIDI file)")
print(f"  EnCodec tokens       {int(sec * 200):>10,}        REMI-style tokens (bar, position, pitch, duration): {len(notes) * 3 + length16 // 16:,}")

# ---------- 3. all 371 ----------
tot_notes = sum(len(x["notes"]) for x in ch); tot_sec = sum(max(s + d for s, d, _ in x["notes"]) for x in ch) * SIXTEENTH
print(f"\n== 3. All {len(ch)} chorales: {tot_notes:,} notes, {tot_sec / 60:.0f} minutes of music at {BPM} BPM")
json.dump({"phrase": {"midi_like": midi_like, "remi": remi, "compound": compound, "abc": abc, "rows": rows},
           "chorale": {"title": c["title"], "notes": len(notes), "seconds": round(sec, 1), "midi_bytes": mb, "wav_bytes": wav, "mp3_bytes": mp3},
           "corpus": {"chorales": len(ch), "notes": tot_notes, "minutes": round(tot_sec / 60, 1)}},
          open(os.path.join(HERE, "by_hand.json"), "w"), indent=1)
