"""Measured: how time and memory grow with sequence length, for one attention layer and one state space layer.

Both layers are 256 wide. Each measurement runs in a fresh process, forward pass only, on the CPU.
  attention  torch's scaled dot-product attention with 4 heads and a causal mask
  state space  the block in ssm.py

Run:  python scaling.py        (about 5 minutes; the longest attention run needs about 8 GB of RAM)
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = 256

def one(kind, n):
    import resource, torch, torch.nn as nn
    torch.manual_seed(0); x = torch.randn(1, n, D)
    if kind == "attention":
        layer = nn.MultiheadAttention(D, 4, batch_first=True)
        f = lambda: layer(x, x, x, attn_mask=torch.triu(torch.ones(n, n, dtype=torch.bool), 1), need_weights=False)[0]
    else:
        from ssm import SelectiveSSM
        layer = SelectiveSSM(D); f = lambda: layer(x)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.no_grad():
        t0 = time.time(); f(); dt = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1e6 if sys.platform == "darwin" else 1e3                             # bytes on macOS, kB on Linux
    print("RESULT " + json.dumps({"kind": kind, "n": n, "seconds": dt, "extra_mb": (peak - base) / unit}))

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one": one(sys.argv[2], int(sys.argv[3])); sys.exit()
    res = []
    print(f"{'frames':>8s} {'= music at 50/s':>16s} | {'attention':>22s} | {'state space':>22s}")
    for n in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        row = {}
        for kind in ("attention", "state space"):
            if kind == "attention" and n > 32768: continue                      # 65,536 frames would need about 28 GB for the attention
            out = subprocess.run([sys.executable, __file__, "--one", kind, str(n)], capture_output=True, text=True, cwd=HERE)
            line = next((l for l in out.stdout.split("\n") if l.startswith("RESULT ")), None)
            if line: row[kind] = json.loads(line[7:]); res.append(row[kind])
        cell = lambda k: f"{row[k]['seconds']:>8.2f} s {row[k]['extra_mb']:>8,.0f} MB" if k in row else f"{'not run':>22s}"
        print(f"{n:>8,} {n / 50 / 60:>12.1f} min | {cell('attention')} | {cell('state space')}", flush=True)
    json.dump(res, open(os.path.join(HERE, "scaling.json"), "w"), indent=1)
