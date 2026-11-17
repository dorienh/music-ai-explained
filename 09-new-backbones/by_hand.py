"""Attention's bill and a state's memory, with small numbers.

Run:  python by_hand.py
"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))

print("== 1. Attention compares every frame with every other frame")
rows = []
for label, sec, fps in (("MusicGen clip, 30 s", 30, 50), ("3-minute song", 180, 50), ("10-minute piece", 600, 50), ("1 hour", 3600, 50)):
    n = sec * fps; pairs = n * n; mb = pairs * 4 / 1e6
    rows.append((label, n, pairs, mb)); print(f"  {label:22s} {n:>8,} frames -> {pairs:>16,} pairs -> {mb:>10,.0f} MB per attention map (4 bytes per pair)")
print("  A state space layer instead carries a fixed-size state from frame to frame: cost grows with n, not n x n.")

print("\n== 2. One number of memory: h_t = a * h_(t-1) + b * x_t, with a = 0.5, b = 1")
h, trace = 0.0, []
for t, x in enumerate([1, 0, 0, 0, 0]):
    h = 0.5 * h + 1 * x; trace.append(h); print(f"  t={t}  input {x}  state {h}")

print("\n== 3. How long does a state remember? The input's share after n steps is a^n; half is gone after ln(0.5)/ln(a) steps")
half = []
for a in (0.5, 0.9, 0.99, 0.999, 0.9999):
    n = math.log(0.5) / math.log(a); half.append((a, n, n / 50)); print(f"  a = {a:<7} half-life {n:>8,.0f} steps = {n / 50:>7.2f} s at 50 frames per second")

print("\n== 4. Selection: let the input set a and b at every step")
print("  rule: on a marked input, a=0 and b=1 (forget the old, store this); on anything else, a=1 and b=0 (keep, ignore)")
h, sel = 0.0, []
for t, (x, marked) in enumerate([(3, False), (7, True), (4, False), (9, False), (2, False), (5, False)]):
    a, b = (0, 1) if marked else (1, 0); h = a * h + b * x; sel.append(h); print(f"  t={t}  input {x}{'*' if marked else ' '}  a={a} b={b}  state {h}")
print("  the marked 7 is still there four steps later, untouched by what came after")
json.dump({"attention": rows, "decay": trace, "half_life": half, "selective": sel}, open(os.path.join(HERE, "by_hand.json"), "w"), indent=1)
