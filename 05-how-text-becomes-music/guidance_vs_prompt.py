"""Does stronger guidance make MusicGen follow the prompt more closely? Scored with CLAP.
Generates 5-second clips for several prompts, seeds and guidance scales (CPU, about 10 minutes)."""
import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import json, time, numpy as np, torch, librosa, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration, ClapModel, ClapProcessor, logging
logging.set_verbosity_error(); torch.set_grad_enabled(False)

mg_proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval(); cproc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
norm = lambda x: torch.nn.functional.normalize(x, dim=-1)
def clap_score(audio32k, texts):
    a = norm(clap.get_audio_features(**cproc(audios=[librosa.resample(audio32k, orig_sr=32000, target_sr=48000)], sampling_rate=48000, return_tensors="pt")))
    t = norm(clap.get_text_features(**cproc(text=texts, return_tensors="pt", padding=True)))
    return (t @ a.T).squeeze(1).tolist()

prompts = ["sad jazz with brushed drums", "fast techno with a heavy kick drum", "solo classical guitar, slow and gentle", "an orchestra playing a triumphant fanfare"]
scales, seeds, rows = [1.0, 3.0, 7.0], [0, 1], []
for pi, prompt in enumerate(prompts):
    inp = mg_proc(text=[prompt], padding=True, return_tensors="pt")
    for scale in scales:
        for seed in seeds:
            torch.manual_seed(seed); t = time.time()
            audio = mg.generate(**inp, do_sample=True, guidance_scale=scale, max_new_tokens=250)[0, 0].numpy()
            s = clap_score(audio, prompts)                       # similarity to every prompt, not only its own
            rows.append(dict(prompt=prompt, scale=scale, seed=seed, own=s[pi], others=float(np.mean([x for k, x in enumerate(s) if k != pi])), top1=int(np.argmax(s) == pi), seconds=time.time() - t))
            if seed == 0: sf.write(f"out/p{pi}_cfg{scale:g}.wav", audio, 32000)
            print(f"{prompt[:28]:28s} scale {scale:3.0f} seed {seed}: own {s[pi]:.3f}  others {rows[-1]['others']:.3f}  best match is own prompt: {bool(rows[-1]['top1'])}  ({rows[-1]['seconds']:.0f} s)", flush=True)
json.dump(rows, open("guidance_vs_prompt.json", "w"), indent=1)
print("\nscale   CLAP score with own prompt   with the other prompts   own prompt ranked first")
for scale in scales:
    r = [x for x in rows if x["scale"] == scale]
    print(f"{scale:5.0f}   {np.mean([x['own'] for x in r]):26.3f}   {np.mean([x['others'] for x in r]):22.3f}   {sum(x['top1'] for x in r)} of {len(r)}")
