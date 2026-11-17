"""What does it take to run an open music model on your own machine? Measured, not quoted.

For each model: parameters per component, memory the weights should need (parameters x 4 bytes),
the peak memory the process really used while generating, and the time for 10 seconds of music.
Each measurement runs in a fresh process so that the peak memory belongs to that model alone.

Run:  python measure_local.py                       (MusicGen-small and -medium on the CPU; about 10 minutes, 6 GB of downloads)
      python measure_local.py facebook/musicgen-small mps      (one model on an Apple GPU)
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT, SECONDS = "upbeat folk pop song with acoustic guitar, bass and drums", 10

def one(name, device):
    import resource, torch, soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(name); model = MusicgenForConditionalGeneration.from_pretrained(name).eval().to(device)
    load = time.time() - t0
    count = lambda m: sum(p.numel() for p in m.parameters())
    parts = {"text encoder (T5)": count(model.text_encoder), "language model": count(model.decoder) + count(model.enc_to_dec_proj), "codec (EnCodec)": count(model.audio_encoder)}
    torch.manual_seed(0); x = proc(text=[PROMPT], return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad(): y = model.generate(**x, do_sample=True, guidance_scale=3, max_new_tokens=SECONDS * 50)[0, 0].cpu().numpy()
    gen = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)       # bytes on macOS, kB on Linux
    gpu = torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0)
    os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
    sf.write(os.path.join(HERE, "audio", f"{name.split('/')[-1]}_{device}.wav"), y, 32000)
    print("RESULT " + json.dumps({"model": name, "device": device, "params": parts, "weights_gb_fp32": sum(parts.values()) * 4 / 1e9,
                                  "peak_ram_gb": peak, "gpu_gb": gpu, "load_s": load, "gen_s": gen, "seconds_per_second": gen / SECONDS}))

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one": one(sys.argv[2], sys.argv[3]); sys.exit()
    jobs = [(sys.argv[1], sys.argv[2])] if len(sys.argv) == 3 else [("facebook/musicgen-small", "cpu"), ("facebook/musicgen-medium", "cpu")]
    path = os.path.join(HERE, "measure_local.json"); res = json.load(open(path)) if os.path.exists(path) else {}
    for name, device in jobs:
        out = subprocess.run([sys.executable, __file__, "--one", name, device], capture_output=True, text=True)
        line = next((l for l in out.stdout.split("\n") if l.startswith("RESULT ")), None)
        if not line: print(name, device, "failed:", out.stderr[-400:]); continue
        r = json.loads(line[7:]); res[f"{name} | {device}"] = r
        print(f"\n{name} on {device}")
        for k, v in r["params"].items(): print(f"   {k:20s} {v / 1e6:>8.0f} M parameters")
        print(f"   weights at 4 bytes each: {r['weights_gb_fp32']:.2f} GB;  peak process memory: {r['peak_ram_gb']:.2f} GB" + (f";  GPU memory: {r['gpu_gb']:.2f} GB" if r["gpu_gb"] else ""))
        print(f"   {SECONDS} s of music in {r['gen_s']:.0f} s = {r['seconds_per_second']:.1f} s of computing per second of music (loading took {r['load_s']:.0f} s)")
    json.dump(res, open(path, "w"), indent=1)
