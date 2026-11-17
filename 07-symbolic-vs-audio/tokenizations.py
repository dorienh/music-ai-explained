"""How many tokens does the same music cost? Six symbolic tokenizations of the Bach chorales, next to an audio codec.

Uses MidiTok (Fradet et al. 2021), which implements the published tokenizations behind one interface.
Run:  python tokenizations.py      (about a minute on a laptop CPU)
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from miditok import MIDILike, TSD, REMI, Structured, CPWord, Octuple, TokenizerConfig
from common import load_chorales, write_midi, OUT, HERE, SIXTEENTH

ch = load_chorales()
mdir = os.path.join(OUT, "chorales_midi"); os.makedirs(mdir, exist_ok=True)
paths = []
for i, c in enumerate(ch):
    p = os.path.join(mdir, f"{i:03d}.mid"); paths.append(p)
    if not os.path.exists(p): write_midi(c["notes"], p)
n_notes = sum(len(c["notes"]) for c in ch)
seconds = sum(max(s + d for s, d, _ in c["notes"]) for c in ch) * SIXTEENTH
print(f"{len(ch)} chorales, {n_notes:,} notes, {seconds / 60:.0f} minutes, {n_notes / seconds:.1f} notes per second\n")

def count(tok):
    steps = tokens = 0
    for p in paths:
        ids = tok(p)[0].ids if isinstance(tok(p), list) else tok(p).ids
        steps += len(ids)
        tokens += sum(len(x) for x in ids) if ids and isinstance(ids[0], list) else len(ids)   # compound tokens carry several fields
    return steps, tokens

cfg = lambda: TokenizerConfig(use_velocities=True, use_tempos=False, use_time_signatures=False, use_programs=False)
res = {}
print(f"{'tokenization':28s} {'vocabulary':>10s} {'steps/note':>11s} {'steps/s':>9s}   what one step is")
for name, cls, what in [("MIDI-like", MIDILike, "note-on, note-off, time-shift or velocity"),
                        ("TSD", TSD, "like MIDI-like, with a duration instead of a note-off"),
                        ("REMI", REMI, "bar, position, pitch, velocity or duration"),
                        ("Structured", Structured, "pitch, velocity, duration, time-shift in fixed order"),
                        ("Compound Word", CPWord, "one bundle of fields per note or per position"),
                        ("Octuple", Octuple, "one bundle of fields per note")]:
    tok = cls(cfg()); steps, fields = count(tok)
    vocab = sum(len(v) for v in tok.vocab) if isinstance(tok.vocab, list) else len(tok.vocab)
    res[name] = {"vocab": vocab, "steps": steps, "fields": fields, "per_note": steps / n_notes, "per_second": steps / seconds, "what": what}
    print(f"{name:28s} {vocab:>10,} {steps / n_notes:>11.2f} {steps / seconds:>9.1f}   {what}")

# byte-pair encoding on top of REMI: merge token pairs that often occur together into one new token
for size in (1000, 5000):
    tok = REMI(cfg()); base = len(tok.vocab)
    tok.train(vocab_size=size, model="BPE", files_paths=paths)
    steps, _ = count(tok)
    res[f"REMI + BPE ({size})"] = {"vocab": size, "steps": steps, "per_note": steps / n_notes, "per_second": steps / seconds, "what": "a learned group of REMI tokens"}
    print(f"{'REMI + BPE':28s} {size:>10,} {steps / n_notes:>11.2f} {steps / seconds:>9.1f}   learned groups of REMI tokens (base vocabulary {base})")

print("\nAudio, for comparison (fixed rates, independent of how many notes are played):")
for name, steps_s, tok_s, vocab in [("EnCodec 32 kHz (MusicGen)", 50, 200, 2048), ("DAC 44.1 kHz", 86, 774, 1024), ("Stable Audio latent (64 numbers per frame)", 21.5, 1376, None)]:
    res[name] = {"per_second": steps_s, "tokens_per_second": tok_s, "vocab": vocab, "audio": True}
    print(f"  {name:44s} {steps_s:>6} frames/s, {tok_s:>6} numbers or codebook indices per second")
res["_corpus"] = {"notes": n_notes, "seconds": seconds}
json.dump(res, open(os.path.join(HERE, "tokenizations.json"), "w"), indent=1)
