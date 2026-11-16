import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, math, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "out"
BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "Helvetica Neue",
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2})
np.set_printoptions(precision=3, suppress=True)

print("== A. sampling arithmetic")
logits = np.array([2.0, 1.0, 0.5, -1.0])
sm = lambda l: np.exp(l - l.max())/np.exp(l - l.max()).sum()
for T in [0.5, 1.0, 2.0]: print("T", T, sm(logits/T))
p = sm(logits); k2 = p.copy(); k2[2:] = 0; print("top-2", k2/k2.sum())
cond = np.array([2.0, 1.0, 0.5, -1.0]); unc = np.array([1.0, 1.0, 1.0, 0.0])
for s in [1, 3, 10]: print("cfg", s, "logits", unc + s*(cond-unc), "probs", sm(unc + s*(cond-unc)))
print("uncond probs", sm(unc))

print("== B. toy 2D: diffusion (DDIM sampling) vs flow matching (Euler)")
torch.manual_seed(0); np.random.seed(0)
def data(n):
    k = torch.randint(0, 8, (n,)); ang = k * (2*math.pi/8)
    return torch.stack([2*torch.cos(ang), 2*torch.sin(ang)], 1) + 0.1*torch.randn(n, 2)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(2+16, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    def forward(s, x, t):
        fr = 2**torch.arange(8, dtype=torch.float32) * math.pi
        e = torch.cat([torch.sin(t[:, None]*fr), torch.cos(t[:, None]*fr)], 1)
        return s.f(torch.cat([x, e], 1))
STEPS, BS = 20000, 1024
# diffusion: variance-preserving, cosine schedule, noise prediction
abar = lambda t: torch.cos(t * math.pi/2)**2   # t in [0,1], 0 = clean
dm = Net(); opt = torch.optim.Adam(dm.parameters(), 1e-3)
for i in range(STEPS):
    x1 = data(BS); t = torch.rand(BS); eps = torch.randn(BS, 2); a = abar(t)[:, None]
    v = a.sqrt()*eps - (1-a).sqrt()*x1                      # v-prediction target
    loss = ((dm(a.sqrt()*x1 + (1-a).sqrt()*eps, t) - v)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("diffusion final loss", loss.item())
fm = Net(); opt = torch.optim.Adam(fm.parameters(), 1e-3)
for i in range(STEPS):
    x1 = data(BS); x0 = torch.randn(BS, 2); t = torch.rand(BS)
    xt = (1-t[:, None])*x0 + t[:, None]*x1
    loss = ((fm(xt, t) - (x1 - x0))**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("flow matching final loss", loss.item())

# naive diffusion: noise prediction, sampling started at t=0.999 (the original DDPM parameterisation)
de = Net(); opt = torch.optim.Adam(de.parameters(), 1e-3)
for i in range(STEPS):
    x1 = data(BS); t = torch.rand(BS)*0.999; eps = torch.randn(BS, 2); a = abar(t)[:, None]
    loss = ((de(a.sqrt()*x1 + (1-a).sqrt()*eps, t) - eps)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def sample_eps(x, N):
    ts = torch.linspace(0.999, 0, N+1)
    for i in range(N):
        t, tn = ts[i], ts[i+1]; a, an = abar(t), abar(tn)
        eps = de(x, t.expand(len(x))); x0 = (x - (1-a).sqrt()*eps)/a.sqrt()
        x = an.sqrt()*x0 + (1-an).sqrt()*eps
    return x
@torch.no_grad()
def sample_ddim(x, N, traj=False):
    ts = torch.linspace(1.0, 0, N+1); path = [x.clone()]
    for i in range(N):
        t, tn = ts[i], ts[i+1]; a, an = abar(t).clamp(0, 1), abar(tn).clamp(0, 1)
        v = dm(x, t.expand(len(x)))
        x0 = a.sqrt()*x - (1-a).sqrt()*v; eps = (1-a).sqrt()*x + a.sqrt()*v
        x = an.sqrt()*x0 + (1-an).sqrt()*eps
        path.append(x.clone())
    return (x, torch.stack(path)) if traj else x
@torch.no_grad()
def sample_fm(x, N, traj=False):
    path = [x.clone()]; dt = 1.0/N
    for i in range(N):
        x = x + dt*fm(x, torch.full((len(x),), i*dt)); path.append(x.clone())
    return (x, torch.stack(path)) if traj else x
def swd(a, b, n_proj=256):
    g = torch.Generator().manual_seed(1); th = torch.randn(n_proj, 2, generator=g); th /= th.norm(dim=1, keepdim=True)
    pa, pb = (a @ th.T).sort(0).values, (b @ th.T).sort(0).values
    return (pa - pb).abs().mean().item()
centers = torch.stack([2*torch.cos(torch.arange(8)*2*math.pi/8), 2*torch.sin(torch.arange(8)*2*math.pi/8)], 1)
def on_mode(x): return ((x[:, None] - centers[None]).norm(dim=2).min(1).values < 0.3).float().mean().item()
real = data(10000); real2 = data(10000)
print("floor: real vs real SWD %.4f, on-mode %.3f" % (swd(real, real2), on_mode(real2)))
noise = torch.randn(10000, 2); res = {"steps": [], "ddim": [], "fm": [], "ddim_mode": [], "fm_mode": [], "eps": [], "eps_mode": []}
for N in [1, 2, 4, 8, 16, 32, 64, 128]:
    a, b = sample_ddim(noise.clone(), N), sample_fm(noise.clone(), N)
    res["steps"].append(N); res["ddim"].append(swd(a, real)); res["fm"].append(swd(b, real)); res["ddim_mode"].append(on_mode(a)); res["fm_mode"].append(on_mode(b))
    c = sample_eps(noise.clone(), N); res["eps"].append(swd(c, real)); res["eps_mode"].append(on_mode(c))
    print(f"      naive eps-pred diffusion SWD {res['eps'][-1]:.4f} on-mode {100*res['eps_mode'][-1]:5.1f}%")
    print(f"N={N:3d}  diffusion SWD {res['ddim'][-1]:.4f} on-mode {100*res['ddim_mode'][-1]:5.1f}%   flow SWD {res['fm'][-1]:.4f} on-mode {100*res['fm_mode'][-1]:5.1f}%")
res["floor"] = swd(real, real2); json.dump(res, open(f"{OUT}/toy_steps.json", "w"))
# straightness: path length / straight-line distance, 200 steps
z = torch.randn(2000, 2)
_, pd = sample_ddim(z.clone(), 200, True); _, pf = sample_fm(z.clone(), 200, True)
def straight(p): return ((p[1:]-p[:-1]).norm(dim=2).sum(0) / (p[-1]-p[0]).norm(dim=1)).median().item()
def dev(p):
    d = p[-1]-p[0]; L = d.norm(dim=1); u = d/L[:, None]; r = p - p[0]
    perp = r - (r*u).sum(2, keepdim=True)*u
    return (perp.norm(dim=2).max(0).values / L).median().item()
print("median max deviation from straight line / distance: diffusion %.3f, flow %.3f" % (dev(pd), dev(pf)))
print("median path length / straight distance: diffusion %.3f, flow %.3f" % (straight(pd), straight(pf)))

fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
for ax, p, title, col in [(axs[0], pd, "Diffusion (v-prediction)", BLUE), (axs[1], pf, "Flow matching", ORANGE)]:
    ax.scatter(real[:1500, 0], real[:1500, 1], s=3, color=GRID, zorder=1)
    for j in range(40): ax.plot(p[:, j, 0], p[:, j, 1], color=col, lw=1, alpha=0.75, zorder=2)
    ax.scatter(p[0, :40, 0], p[0, :40, 1], s=14, color=INK2, zorder=3); ax.scatter(p[-1, :40, 0], p[-1, :40, 1], s=14, color=col, zorder=3)
    ax.set_title(title, loc="left", fontsize=11); ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
axs[0].annotate("grey dots: noise starting points\ncoloured dots: where they end", (-3.1, -3.1), fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_paths.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(res["steps"], res["eps"], "-o", color="#1baf7a", ms=8, mec=SURF, mew=2, label="Diffusion, noise prediction (naive setup)")
ax.plot(res["steps"], res["ddim"], "-o", color=BLUE, ms=8, mec=SURF, mew=2, label="Diffusion, v-prediction")
ax.plot(res["steps"], res["fm"], "-o", color=ORANGE, ms=8, mec=SURF, mew=2, label="Flow matching")
ax.axhline(res["floor"], color=INK2, lw=1, ls=(0, (4, 3))); ax.annotate("real data vs real data", (40, res["floor"]), xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=9)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(res["steps"]); ax.set_xticklabels(res["steps"])
ax.set_ylim(0.01, 3); ax.set_xlabel("Number of sampling steps"); ax.set_ylabel("Distance to the real data (lower is better)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_steps.png", dpi=200); plt.close(fig)
