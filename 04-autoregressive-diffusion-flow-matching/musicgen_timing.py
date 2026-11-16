import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import time, torch, numpy as np, soundfile as sf, json
from transformers import AutoProcessor, MusicgenForConditionalGeneration
OUT = "out"
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
sr = model.config.audio_encoder.sampling_rate
print("sr", sr, "frame rate", model.config.audio_encoder.frame_rate, "codebooks", model.decoder.num_codebooks, "vocab", model.config.decoder.vocab_size)
inp = proc(text=["lo-fi beat with warm electric piano"], padding=True, return_tensors="pt")
res = []
def run(tag, **kw):
    torch.manual_seed(0); t = time.time()
    with torch.no_grad(): a = model.generate(**inp, **kw)
    dt = time.time() - t; y = a[0, 0].numpy(); sf.write(f"{OUT}/musicgen_{tag}.wav", y, sr)
    print(f"{tag}: {len(y)/sr:.1f} s audio in {dt:.1f} s  ({kw})", flush=True); return y, dt
for n in [250, 500, 1000]:
    y, dt = run(f"{n}tok", do_sample=True, max_new_tokens=n, guidance_scale=3.0); res.append((n, len(y)/sr, dt))
for g in [1.0, 10.0]:
    run(f"cfg{g:g}", do_sample=True, max_new_tokens=500, guidance_scale=g)
run("greedy", do_sample=False, max_new_tokens=500, guidance_scale=3.0)
run("topk50_t0.5", do_sample=True, max_new_tokens=500, guidance_scale=3.0, top_k=50, temperature=0.5)
print("gen config defaults: top_k", model.generation_config.top_k, "temp", model.generation_config.temperature, "guidance", model.generation_config.guidance_scale, "do_sample", model.generation_config.do_sample)
# token repetition: re-encode outputs with the model's own codec and count unique first-codebook tokens
for tag in ["500tok", "greedy", "cfg1", "cfg10", "topk50_t0.5"]:
    y, _ = sf.read(f"{OUT}/musicgen_{tag}.wav"); x = torch.from_numpy(y).float()[None, None]
    with torch.no_grad(): codes = model.audio_encoder.encode(x).audio_codes[0, 0]   # [K, T]
    c0 = codes[0].numpy(); rep = (c0[1:] == c0[:-1]).mean()
    print(f"{tag}: first-codebook tokens {len(c0)}, unique {len(np.unique(c0))}, immediate repeats {100*rep:.1f}%")
json.dump(res, open(f"{OUT}/musicgen_timing.json", "w"))
