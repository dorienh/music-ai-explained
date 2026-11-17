"""The same request to a symbolic model and to an audio model: what comes out, how big is it, how long does it take?

Symbolic: text2midi (Bhandari et al., AAAI 2025; from my lab), which writes a MIDI file.
Audio:    MusicGen-small (Copet et al., 2023), which writes codec tokens that decode to a waveform.

Setup:  git clone https://github.com/AMAAI-Lab/text2midi ../../text2midi  &&  pip install -r ../../text2midi/requirements.txt
Run:    python text_to_notes_vs_text_to_sound.py [path to the text2midi folder] [tokens to generate, default 800]
        On a laptop CPU this takes several minutes: text2midi's reference code recomputes the whole sequence for every new token.
"""
import json, os, pickle, subprocess, sys, time
import numpy as np, torch, torch.nn as nn, soundfile as sf
from common import OUT, HERE

T2M = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "text2midi")
N_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 800
PROMPT = "A cheerful pop song with piano, electric bass and drums, in C major, 4/4 time, at a moderate tempo of 110 beats per minute."
sys.path.insert(0, T2M)
torch.manual_seed(0)

# ---------- symbolic ----------
from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer
from model.transformer_model import Transformer
remi = pickle.load(open(hf_hub_download("amaai-lab/text2midi", "vocab_remi.pkl"), "rb"))
model = Transformer(len(remi), 768, 8, 2048, 18, 1024, False, 8, device="cpu")
model.load_state_dict(torch.load(hf_hub_download("amaai-lab/text2midi", "pytorch_model.bin"), map_location="cpu")); model.eval()
p_sym = sum(p.numel() for p in model.parameters())
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
x = tok(PROMPT, return_tensors="pt", padding=True, truncation=True)
t0 = time.time()
with torch.no_grad(): ids = model.generate(x.input_ids, x.attention_mask, max_len=N_TOK, temperature=1.0)[0].tolist()
t_sym = time.time() - t0
score = remi.decode(ids); mid_path = os.path.join(HERE, "audio", "text2midi_pop.mid"); os.makedirs(os.path.dirname(mid_path), exist_ok=True)
score.dump_midi(mid_path)
import pretty_midi
pm = pretty_midi.PrettyMIDI(mid_path); dur = pm.get_end_time(); n_notes = sum(len(i.notes) for i in pm.instruments)
tracks = [("drums" if i.is_drum else pretty_midi.program_to_instrument_name(i.program)) for i in pm.instruments]
pitched = [n.pitch % 12 for i in pm.instruments if not i.is_drum for n in i.notes]
in_c = float(np.mean([p in (0, 2, 4, 5, 7, 9, 11) for p in pitched])); tempo = float(pm.get_tempo_changes()[1][0])
print(f"text2midi: {p_sym / 1e6:.0f} M parameters in total (text encoder and decoder), vocabulary {len(remi):,}")
print(f"  asked for C major at 110 BPM: {100 * in_c:.0f}% of the pitched notes are in C major, tempo written in the file {tempo:.0f} BPM")
print(f"  {N_TOK} tokens in {t_sym:.0f} s -> {n_notes} notes, {dur:.1f} s of music, {os.path.getsize(mid_path):,} bytes")
print(f"  {N_TOK / dur:.0f} tokens per second of music; tracks: {', '.join(tracks)}")

sf2 = next((p for p in ("/opt/homebrew/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2", "/usr/share/sounds/sf2/FluidR3_GM.sf2") if os.path.exists(p)), None)
if sf2:                                                      # the MIDI file is silent until some instrument plays it
    wav = os.path.join(HERE, "audio", "text2midi_pop.wav")
    subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-F", wav, "-r", "32000", sf2, mid_path], capture_output=True)
    print("  rendered with FluidSynth and", os.path.basename(sf2))

# ---------- audio ----------
from transformers import AutoProcessor, MusicgenForConditionalGeneration
proc = AutoProcessor.from_pretrained("facebook/musicgen-small"); mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
p_aud = sum(p.numel() for p in mg.parameters())
secs = min(30, dur); frames = int(secs * 50)
t0 = time.time()
with torch.no_grad(): y = mg.generate(**proc(text=[PROMPT], return_tensors="pt"), do_sample=True, guidance_scale=3, max_new_tokens=frames)[0, 0].numpy()
t_aud = time.time() - t0
sf.write(os.path.join(HERE, "audio", "musicgen_pop.wav"), y, 32000)
print(f"\nMusicGen-small: {p_aud / 1e6:.0f} M parameters in total (text encoder, language model and codec)")
print(f"  {frames} frames x 4 codebooks = {frames * 4:,} tokens in {t_aud:.0f} s -> {len(y) / 32000:.1f} s of music, {len(y) * 2:,} bytes as 16-bit audio")
print(f"  200 tokens per second of music")
print(f"\nPer second of music: text2midi {N_TOK / dur:.0f} tokens, MusicGen 200 tokens ({200 / (N_TOK / dur):.1f} x more)")
json.dump({"prompt": PROMPT, "symbolic": {"params": p_sym, "tokens": N_TOK, "seconds_compute": t_sym, "notes": n_notes, "seconds_music": dur, "bytes": os.path.getsize(mid_path), "tracks": tracks, "in_c_major": in_c, "tempo": tempo, "vocab": len(remi)},
           "audio": {"params": p_aud, "tokens": frames * 4, "seconds_compute": t_aud, "seconds_music": len(y) / 32000, "bytes": len(y) * 2}},
          open(os.path.join(HERE, "text_to_notes_vs_text_to_sound.json"), "w"), indent=1)
