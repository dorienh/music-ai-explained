"""Make a 90-second piece with MusicGen the way people stretch short-context models:
generate 30 s, then repeatedly feed the last 10 s back in as an audio prompt and
generate 20 s more. The model only ever sees 30 seconds at a time.

Run:  python generate_long.py [seed ...]     (about 10 minutes per seed on a laptop CPU)
Writes out/musicgen_90s_s<seed>.wav. The article uses seeds 0, 1 and 2.
"""
import os, sys, time
import numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
PROMPT = "upbeat folk pop song with acoustic guitar, bass, drums and a catchy chorus"
SR, WINDOW, KEEP, TOTAL = 32000, 30, 10, 90

SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
tokens = lambda s: int(s * 50)                            # 50 codec frames per second

for seed in SEEDS:
    torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        x = proc(text=[PROMPT], return_tensors="pt")
        song = model.generate(**x, do_sample=True, guidance_scale=3, max_new_tokens=tokens(WINDOW))[0, 0].numpy()
        while len(song) < TOTAL * SR:
            ctx = song[-KEEP * SR:]
            x = proc(text=[PROMPT], audio=[ctx], sampling_rate=SR, return_tensors="pt")
            y = model.generate(**x, do_sample=True, guidance_scale=3, max_new_tokens=tokens(WINDOW - KEEP))[0, 0].numpy()
            song = np.concatenate([song, y[len(ctx):]])        # output starts with the prompt audio
            print(f"{len(song)/SR:.0f} s generated, {time.time()-t0:.0f} s elapsed", flush=True)
    sf.write(os.path.join(OUT, f"musicgen_90s_s{seed}.wav"), song[: TOTAL * SR], SR)
    print("done in", round(time.time() - t0), "s")
