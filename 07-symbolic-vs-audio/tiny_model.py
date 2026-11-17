"""A symbolic music model small enough to train on a laptop: a transformer on 371 Bach chorales.

Tokens: BAR, POS_0..15 (sixteenth within the bar), PITCH_p, DUR_d (sixteenths). All chorales are moved to C major or A minor.
Compared against two baselines (a bigram table and random tokens) and against held-out real chorales, with four checks:
  grammar   share of tokens that are legal after the token before them
  in scale  share of notes that fit the best-fitting major scale
  triads    share of beats on which the sounding notes form a major or minor triad
  copied    share of bars that occur note for note in the training chorales

Run:  python tiny_model.py [steps]      (default 2000 steps, about 10 minutes on a laptop CPU)
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, soundfile as sf
from common import load_chorales, write_midi, synth, to_seconds, OUT, HERE

torch.manual_seed(0); rng = np.random.default_rng(0)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CTX, D, LAYERS, HEADS, BATCH = 256, 128, 4, 4, 32

# ---------- tokens ----------
LO, HI = 30, 90                                                      # pitch range kept
VOCAB = ["BAR"] + [f"POS_{i}" for i in range(16)] + [f"PITCH_{p}" for p in range(LO, HI)] + [f"DUR_{d}" for d in range(1, 17)]
ID = {t: i for i, t in enumerate(VOCAB)}
kind = lambda i: VOCAB[i].split("_")[0]

def encode(c):
    shift = (-c["tonic"] if c["mode"] == "major" else 9 - c["tonic"]) % 12
    if shift > 6: shift -= 12                                        # move to C major / A minor by the shortest route
    out, last_bar, last_pos = [], -1, None
    for s, d, p in sorted(c["notes"], key=lambda n: (n[0], -n[2])):
        bar, pos = divmod(s, 16)
        while last_bar < bar: out.append(ID["BAR"]); last_bar += 1; last_pos = None
        if pos != last_pos: out.append(ID[f"POS_{pos}"]); last_pos = pos
        out += [ID[f"PITCH_{min(HI - 1, max(LO, p + shift))}"], ID[f"DUR_{min(16, d)}"]]
    return out

def decode(ids):
    """Tokens -> notes. Returns (notes, number of tokens that broke the grammar)."""
    notes, bar, pos, pitch, bad, prev = [], -1, None, None, 0, None
    for i in ids:
        k, v = kind(i), VOCAB[i].split("_")[-1]
        legal = {"BAR": prev in (None, "BAR", "DUR"), "POS": prev in ("BAR", "DUR"), "PITCH": prev in ("POS", "DUR"), "DUR": prev == "PITCH"}[k]
        if k == "POS" and legal and prev == "DUR" and int(v) <= pos: legal = False     # positions must move forward inside a bar
        if not legal: bad += 1; continue
        if k == "BAR": bar += 1; pos = None
        elif k == "POS": pos = int(v)
        elif k == "PITCH": pitch = int(v)
        else: notes.append((bar * 16 + pos, int(v), pitch))
        prev = k
    return notes, bad

ch = load_chorales(); order = rng.permutation(len(ch)); n_val = len(ch) // 10
val = [encode(ch[i]) for i in order[:n_val]]; train = [encode(ch[i]) for i in order[n_val:]]
n_tok = sum(map(len, train))
print(f"{len(train)} training chorales ({n_tok:,} tokens), {len(val)} held out ({sum(map(len, val)):,} tokens), vocabulary {len(VOCAB)}")

# ---------- baselines ----------
V = len(VOCAB)
uni = np.ones(V); bi = np.ones((V, V))
for s in train:
    for a in s: uni[a] += 1
    for a, b in zip(s, s[1:]): bi[a, b] += 1
uni /= uni.sum(); bi /= bi.sum(1, keepdims=True)
nll = lambda f: float(np.mean([f(s) for s in val]))
uni_loss = -np.mean(np.concatenate([np.log(uni[s[1:]]) for s in val]))
bi_loss = -np.mean(np.concatenate([np.log(bi[s[:-1], s[1:]]) for s in val]))
print(f"held-out loss (nats per token; lower is better): guessing uniformly {np.log(V):.2f}, unigram {uni_loss:.2f}, bigram {bi_loss:.2f}")

# ---------- the model ----------
class Tiny(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.1, batch_first=True, norm_first=True)
        s.body, s.norm, s.head = nn.TransformerEncoder(layer, LAYERS), nn.LayerNorm(D), nn.Linear(D, V)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.full((T, T), float("-inf")), 1)       # each position sees only earlier ones
        return s.head(s.norm(s.body(s.tok(x) + s.pos(torch.arange(T)), mask=mask)))

model = Tiny(); n_par = sum(p.numel() for p in model.parameters())
print(f"model: {n_par:,} parameters, {LAYERS} layers, context {CTX} tokens")
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
def batch(seqs):
    xs = []
    for _ in range(BATCH):
        s = seqs[rng.integers(len(seqs))]; i = rng.integers(0, max(1, len(s) - CTX - 1)); xs.append(s[i:i + CTX + 1])
    n = min(map(len, xs)); x = torch.tensor([q[:n] for q in xs])
    return x[:, :-1], x[:, 1:]
@torch.no_grad()
def val_loss():
    model.eval(); tot = cnt = 0
    for s in val:
        for i in range(0, len(s) - 1, CTX):
            x = torch.tensor(s[i:i + CTX + 1])[None]
            if x.shape[1] < 2: continue
            l = nn.functional.cross_entropy(model(x[:, :-1])[0], x[0, 1:], reduction="sum"); tot += l.item(); cnt += x.shape[1] - 1
    model.train(); return tot / cnt
curve, best, t0 = [], 9, time.time()
for step in range(1, STEPS + 1):
    x, y = batch(train)
    loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 250 == 0 or step == STEPS:
        v = val_loss(); curve.append((step, round(loss.item(), 3), round(v, 3), round(time.time() - t0)))
        if v < best: best = v; torch.save(model.state_dict(), os.path.join(OUT, "tiny.pt"))
        print(f"step {step:5d}  train {loss.item():.3f}  held-out {v:.3f}  {time.time() - t0:.0f} s", flush=True)
train_time = time.time() - t0
model.load_state_dict(torch.load(os.path.join(OUT, "tiny.pt"))); model.eval()
print(f"best held-out loss {best:.3f} nats per token = the model is as unsure as a fair choice among {np.exp(best):.1f} tokens (bigram: {np.exp(bi_loss):.1f}, uniform: {V})")

# ---------- generate ----------
@torch.no_grad()
def sample_model(n):
    x = torch.tensor([[ID["BAR"]]])
    for _ in range(n):
        p = torch.softmax(model(x[:, -CTX:])[0, -1], -1); x = torch.cat([x, torch.multinomial(p, 1)[None]], 1)
    return x[0].tolist()
def sample_bigram(n):
    s = [ID["BAR"]]
    for _ in range(n): s.append(int(rng.choice(V, p=bi[s[-1]])))
    return s
N, LEN = 20, 600
t0 = time.time(); gen = [sample_model(LEN) for _ in range(N)]; gen_time = (time.time() - t0) / N
sets = {"Held-out real chorales": val, "Tiny transformer": gen, "Bigram table": [sample_bigram(LEN) for _ in range(N)],
        "Random tokens": [rng.integers(0, V, LEN).tolist() for _ in range(N)]}

# ---------- checks ----------
MAJOR = {0, 2, 4, 5, 7, 9, 11}
TRIADS = {frozenset((r + a) % 12 for a in iv) for r in range(12) for iv in ((0, 4, 7), (0, 3, 7))}
def bars_of(seq):
    out, cur = [], None
    for i in seq:
        if i == ID["BAR"]:
            if cur: out.append(tuple(cur))
            cur = []
        elif cur is not None: cur.append(i)
    return out
train_bars = {b for s in train for b in bars_of(s) if len(b) >= 8}
def checks(seqs):
    bad = tot = 0; scale, tri, copied = [], [], []
    for s in seqs:
        notes, b = decode(s); bad += b; tot += len(s)
        if len(notes) < 8: continue
        pcs = np.bincount([p % 12 for _, _, p in notes], minlength=12)
        scale.append(max(sum(pcs[(k + r) % 12] for k in MAJOR) for r in range(12)) / pcs.sum())
        hits = beats = 0
        for t in range(0, max(a + d for a, d, _ in notes), 4):                         # every quarter-note beat
            sounding = {p % 12 for a, d, p in notes if a <= t < a + d}
            if len(sounding) >= 3: beats += 1; hits += frozenset(sounding) in TRIADS
        if beats: tri.append(hits / beats)
        bs = [x for x in bars_of(s) if len(x) >= 8]
        if bs: copied.append(np.mean([x in train_bars for x in bs]))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"grammar_ok": 1 - bad / tot, "in_scale": f(scale), "triads": f(tri), "copied_bars": f(copied)}
res = {k: checks(v) for k, v in sets.items()}
print(f"\n{'':24s} {'grammar ok':>11s} {'in scale':>9s} {'triads on beats':>16s} {'bars copied':>12s}")
for k, r in res.items(): print(f"{k:24s} {100 * r['grammar_ok']:>10.1f}% {100 * r['in_scale']:>8.1f}% {100 * r['triads']:>15.1f}% {100 * r['copied_bars']:>11.1f}%")
print(f"\ntraining took {train_time:.0f} s on the CPU; generating one piece of {LEN} tokens takes {gen_time:.1f} s")

# ---------- listen ----------
adir = os.path.join(HERE, "audio"); os.makedirs(adir, exist_ok=True)
for name, seq in (("chorale_generated", gen[0]), ("chorale_bigram", sets["Bigram table"][0]), ("chorale_real", val[0][:LEN])):
    notes, _ = decode(seq); notes = [n for n in notes if n[0] < 16 * 12]                # first 12 bars
    write_midi(notes, os.path.join(adir, name + ".mid")); sf.write(os.path.join(adir, name + ".wav"), synth(to_seconds(notes), decay=0.8), 22050)
    json.dump(notes, open(os.path.join(OUT, name + ".json"), "w"))
json.dump({"params": n_par, "train_tokens": n_tok, "vocab": V, "steps": STEPS, "train_seconds": round(train_time), "gen_seconds": round(gen_time, 2),
           "loss": {"uniform": float(np.log(V)), "unigram": float(uni_loss), "bigram": float(bi_loss), "transformer": best}, "curve": curve, "checks": res},
          open(os.path.join(HERE, "tiny_model.json"), "w"), indent=1)
