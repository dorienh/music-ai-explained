"""The same task, the same data, two backbones: a small transformer and a small state space model on Bach chorales.

Task and tokens are those of ../07-symbolic-vs-audio/tiny_model.py: predict the next token of a four-part chorale
(BAR, POS_0..15, PITCH_p, DUR_d; all chorales moved to C major or A minor). Both models are 4 layers, 128 wide,
trained for the same number of steps on the same batches of 256-token windows.

Compared: held-out loss, time per training step, the share of beats carrying a triad in generated music,
and one thing only the state space model can do: read a whole chorale at once, far beyond the 256 tokens it was trained on.

Run:  python train_compare.py [steps]     (default 800 steps: about 5 minutes for the transformer, about 45 for the state space model, on a laptop CPU)
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "07-symbolic-vs-audio"))
from common import load_chorales                     # the chorales as note lists (needs music21 on first use)
from ssm import Block

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
CTX, D, LAYERS, BATCH = 256, 128, 4, 32
LO, HI = 30, 90
VOCAB = ["BAR"] + [f"POS_{i}" for i in range(16)] + [f"PITCH_{p}" for p in range(LO, HI)] + [f"DUR_{d}" for d in range(1, 17)]
ID = {t: i for i, t in enumerate(VOCAB)}; V = len(VOCAB); kind = lambda i: VOCAB[i].split("_")[0]

def encode(c):
    shift = (-c["tonic"] if c["mode"] == "major" else 9 - c["tonic"]) % 12
    if shift > 6: shift -= 12
    out, last_bar, last_pos = [], -1, None
    for s, d, p in sorted(c["notes"], key=lambda n: (n[0], -n[2])):
        bar, pos = divmod(s, 16)
        while last_bar < bar: out.append(ID["BAR"]); last_bar += 1; last_pos = None
        if pos != last_pos: out.append(ID[f"POS_{pos}"]); last_pos = pos
        out += [ID[f"PITCH_{min(HI - 1, max(LO, p + shift))}"], ID[f"DUR_{min(16, d)}"]]
    return out

def decode(ids):
    notes, bar, pos, pitch, prev = [], -1, None, None, None
    for i in ids:
        k, v = kind(i), VOCAB[i].split("_")[-1]
        legal = {"BAR": prev in (None, "BAR", "DUR"), "POS": prev in ("BAR", "DUR"), "PITCH": prev in ("POS", "DUR"), "DUR": prev == "PITCH"}[k]
        if k == "POS" and legal and prev == "DUR" and int(v) <= pos: legal = False
        if not legal: continue
        if k == "BAR": bar += 1; pos = None
        elif k == "POS": pos = int(v)
        elif k == "PITCH": pitch = int(v)
        else: notes.append((bar * 16 + pos, int(v), pitch))
        prev = k
    return notes

TRIADS = {frozenset((r + a) % 12 for a in iv) for r in range(12) for iv in ((0, 4, 7), (0, 3, 7))}
def triads(seqs):
    out = []
    for s in seqs:
        notes = decode(s); hits = beats = 0
        if len(notes) < 8: continue
        for t in range(0, max(a + d for a, d, _ in notes), 4):
            sounding = {p % 12 for a, d, p in notes if a <= t < a + d}
            if len(sounding) >= 3: beats += 1; hits += frozenset(sounding) in TRIADS
        if beats: out.append(hits / beats)
    return float(np.mean(out))

class Transformer(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)                # needs to be told where each token is
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.1, batch_first=True, norm_first=True)
        s.body, s.norm, s.head = nn.TransformerEncoder(layer, LAYERS), nn.LayerNorm(D), nn.Linear(D, V)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        return s.head(s.norm(s.body(s.tok(x) + s.pos(torch.arange(T)), mask=mask)))

class StateSpace(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, D)                                               # no position table: order is built into the scan
        s.body, s.norm, s.head = nn.Sequential(*[Block(D) for _ in range(LAYERS)]), nn.LayerNorm(D), nn.Linear(D, V)
    def forward(s, x): return s.head(s.norm(s.body(s.tok(x))))

ch = load_chorales(); order = np.random.default_rng(0).permutation(len(ch)); n_val = len(ch) // 10   # the same split as in folder 07
val = [encode(ch[i]) for i in order[:n_val]]; train = [encode(ch[i]) for i in order[n_val:]]

@torch.no_grad()
def held_out(model, window):
    """Average loss on the held-out chorales, reading `window` tokens at a time (None = the whole chorale at once)."""
    model.eval(); tot = cnt = 0
    for s in val:
        w = window or len(s)
        for i in range(0, len(s) - 1, w):
            x = torch.tensor(s[i:i + w + 1])[None]
            if x.shape[1] < 2: continue
            tot += nn.functional.cross_entropy(model(x[:, :-1])[0], x[0, 1:], reduction="sum").item(); cnt += x.shape[1] - 1
    model.train(); return tot / cnt

@torch.no_grad()
def generate(model, n=320, pieces=8):
    model.eval(); x = torch.full((pieces, 1), ID["BAR"])
    for _ in range(n):
        p = torch.softmax(model(x[:, -CTX:])[:, -1], -1); x = torch.cat([x, torch.multinomial(p, 1)], 1)
    model.train(); return x.tolist()

res = {"steps": STEPS, "real_triads": triads(val)}
for name, cls in (("Transformer", Transformer), ("State space", StateSpace)):
    torch.manual_seed(0); rng = np.random.default_rng(1)                         # same initial seed, same batches for both
    model = cls(); n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    print(f"\n{name}: {n_par:,} parameters", flush=True); curve, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        xs = []
        for _ in range(BATCH):
            s = train[rng.integers(len(train))]; i = rng.integers(0, max(1, len(s) - CTX - 1)); xs.append(s[i:i + CTX + 1])
        n = min(map(len, xs)); x = torch.tensor([q[:n] for q in xs])
        loss = nn.functional.cross_entropy(model(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == STEPS:
            v = held_out(model, CTX); curve.append((step, round(time.time() - t0), round(v, 3)))
            print(f"  step {step:4d}  held-out loss {v:.3f}  {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    r = {"params": n_par, "seconds": secs, "seconds_per_step": secs / STEPS, "curve": curve, "loss_256": held_out(model, CTX), "triads": triads(generate(model))}
    if name == "State space":
        r["loss_whole"] = held_out(model, None)
        print(f"  reading whole chorales at once (average {np.mean([len(s) for s in val]):.0f} tokens, longest {max(map(len, val))}): held-out loss {r['loss_whole']:.3f}")
    print(f"  final: held-out loss {r['loss_256']:.3f}, {r['seconds_per_step']:.2f} s per step, triads on {100 * r['triads']:.1f}% of beats (real chorales: {100 * res['real_triads']:.1f}%)", flush=True)
    res[name] = r
    json.dump(res, open(os.path.join(HERE, "train_compare.json"), "w"), indent=1)
