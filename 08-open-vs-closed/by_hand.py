"""The arithmetic of running a music model yourself: memory from parameter counts, and cost per song.

Every input number is either a published figure (source in the comment) or a price I looked up on the date shown.
Prices change; edit PRICES and rerun.

Run:  python by_hand.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. memory for the weights ----------
BYTES = {"32-bit float": 4, "16-bit float": 2, "8-bit integer": 1, "4-bit integer": 0.5}
MODELS = [  # name, parameters in millions, source
    ("MusicGen-small", 300, "model card"), ("MusicGen-medium", 1500, "model card"), ("MusicGen-large", 3300, "model card"),
    ("Stable Audio Open 1.0", 1000, "model card: 1B"), ("DiffRhythm", 1100, "paper"),
    ("ACE-Step 1.5 (2B DiT + 0.6B planner)", 2600, "repository"), ("ACE-Step 1.5 XL (4B DiT + 4B planner)", 8000, "repository"),
    ("YuE2-3B", 3000, "repository")]
print("== 1. Memory for the weights alone, in GB (parameters x bytes per parameter / 1e9)")
print(f"   {'':40s}" + "".join(f"{k:>15s}" for k in BYTES))
mem = {}
for name, m, _ in MODELS:
    mem[name] = {k: m * 1e6 * b / 1e9 for k, b in BYTES.items()}
    print(f"   {name:40s}" + "".join(f"{v:>15.1f}" for v in mem[name].values()))

# ---------- 2. memory for attention while generating ----------
print("\n== 2. One attention map: frames x frames x 4 bytes (32-bit), per head, per layer")
for label, n in (("MusicGen, 30 s at 50 frames/s", 1500), ("3-minute song at 50 frames/s", 9000), ("4:45 song at 21.5 frames/s (DiffRhythm)", 6144)):
    print(f"   {label:42s} {n:>6,} x {n:>6,} x 4 = {n * n * 4 / 1e6:>8,.0f} MB")

# ---------- 3. cost per song ----------
PRICES = {"date": "2026-11", "subscription_month": 10.0, "credits_month": 2500, "credits_per_generation": 10, "songs_per_generation": 2,
          "cloud_gpu_hour": {"RTX 4090 (24 GB)": 0.89, "A100 (80 GB)": 2.09},      # RunPod secure cloud, on demand, November 2026
          "electricity_kwh": 0.30, "gpu_watts": 350, "gpu_price": 1800, "gpu_years": 3}
sub = PRICES["subscription_month"] / (PRICES["credits_month"] / PRICES["credits_per_generation"] * PRICES["songs_per_generation"])
print(f"\n== 3. Cost per song")
print(f"   Subscription: ${PRICES['subscription_month']:.0f} / ({PRICES['credits_month']} credits / {PRICES['credits_per_generation']} per generation x {PRICES['songs_per_generation']} songs) = ${sub:.3f} per song, if you use every credit")
cost = {"subscription": sub}
for label, seconds in (("a fast model: 10 s of GPU time per song", 10), ("a slow model: 36 min of GPU time per song", 2160)):   # 36 min: YuE v1's 360 s per 30 s of audio on an RTX 4090, for 3 minutes
    for gpu, rate in PRICES["cloud_gpu_hour"].items():
        c = rate / 3600 * seconds; cost[f"{label} | {gpu}"] = c
        print(f"   Rented {gpu:16s} at ${rate:.2f}/h, {label}: ${c:.4f} per song")
    kwh = PRICES["gpu_watts"] / 1000 * seconds / 3600
    print(f"   Own GPU, electricity only: {PRICES['gpu_watts']} W x {seconds} s = {kwh * 1000:.1f} Wh = ${kwh * PRICES['electricity_kwh']:.4f} per song")
per_month = PRICES["gpu_price"] / (PRICES["gpu_years"] * 12)
print(f"   Own GPU, hardware: ${PRICES['gpu_price']} over {PRICES['gpu_years']} years = ${per_month:.0f} a month, whether you generate or not")
print(f"   Break-even against the subscription: ${per_month:.0f} / ${sub:.3f} = {per_month / sub:,.0f} songs a month")
json.dump({"memory_gb": mem, "prices": PRICES, "cost_per_song": cost, "gpu_month": per_month, "break_even_songs": per_month / sub},
          open(os.path.join(HERE, "by_hand.json"), "w"), indent=1)
