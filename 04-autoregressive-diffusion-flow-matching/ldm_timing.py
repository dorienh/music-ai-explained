import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import time, torch, numpy as np, soundfile as sf, librosa, json, sys
from diffusers import AudioLDM2Pipeline
OUT = "out"; dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32).to(dev)
pipe.set_progress_bar_config(disable=True)
print("device", dev, "scheduler", type(pipe.scheduler).__name__, "unet params (M)", round(sum(p.numel() for p in pipe.unet.parameters())/1e6))
prompt = "lo-fi beat with warm electric piano"
def run(steps, dur, tag=None):
    g = torch.Generator("cpu").manual_seed(0); t = time.time()
    a = pipe(prompt, num_inference_steps=steps, audio_length_in_s=dur, generator=g, guidance_scale=3.5).audios[0]
    dt = time.time() - t
    S = np.abs(librosa.stft(a.astype(np.float32), n_fft=1024))**2; fr = np.fft.rfftfreq(1024, 1/16000)
    hf = S[fr >= 4000].sum()/S.sum(); cen = float((fr[:, None]*S).sum()/S.sum())
    flat = float(librosa.feature.spectral_flatness(y=a.astype(np.float32)).mean())
    if tag: sf.write(f"{OUT}/audioldm2_{tag}.wav", a, 16000)
    print(f"steps {steps:3d} dur {dur:4.1f}s: {dt:6.1f} s  energy above 4 kHz {100*hf:5.2f}%  centroid {cen:6.0f} Hz  flatness {flat:.4f} rms {np.sqrt((a**2).mean()):.3f}", flush=True)
    return dt
run(2, 5.0)  # warm-up
out = {"steps": {}, "dur": {}}
for s in [5, 10, 25, 50, 100, 200]: out["steps"][s] = run(s, 10.0, f"{s}steps")
for d in [5.0, 10.0, 20.0]: out["dur"][d] = run(50, d)
json.dump(out, open(f"{OUT}/ldm_timing.json", "w"))
