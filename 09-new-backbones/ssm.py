"""A minimal selective state space block (the core of Mamba, Gu & Dao 2023) in plain PyTorch.

For each time step t and each channel, the block keeps a small state h (N numbers) and updates it as
    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t          y_t = C_t . h_t
A is learned and fixed; dt_t, B_t and C_t are computed from the input at step t. That is the "selective" part:
the input decides how much of the old state to keep (exp(dt*A) near 1) and how much of itself to write (dt*B).

The scan over time runs in Python, a few steps per pass (chunk=8) or one step per pass (chunk=1, the recurrence as written).
Both give the same numbers; the test at the bottom checks it. This file is for reading and measuring.
The official implementation runs the same recurrence in a fused GPU kernel.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d, expand=2, state=16, conv=4, chunk=8):
        super().__init__()
        self.di, self.N, self.rank, self.chunk = d * expand, state, math.ceil(d / 16), chunk
        self.inp = nn.Linear(d, 2 * self.di, bias=False)                       # two branches: the signal x and a gate z
        self.conv = nn.Conv1d(self.di, self.di, conv, padding=conv - 1, groups=self.di)   # a short look-back before the scan
        self.to_params = nn.Linear(self.di, self.rank + 2 * state, bias=False)  # from x_t: dt (low rank), B_t, C_t
        self.dt = nn.Linear(self.rank, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1).float()).repeat(self.di, 1))   # A = -exp(A_log) < 0
        self.D = nn.Parameter(torch.ones(self.di))
        self.out = nn.Linear(self.di, d, bias=False)

    def forward(self, u):                                                       # u: [batch, time, d]
        T = u.shape[1]
        x, z = self.inp(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))       # causal: only earlier steps
        dt, B, C = self.to_params(x).split([self.rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt(dt))                                            # [batch, time, di], positive
        A = -torch.exp(self.A_log)                                              # [di, N]
        h = torch.zeros(u.shape[0], self.di, self.N, device=u.device); ys = []
        for i in range(0, T, self.chunk):                                       # the scan, a few time steps per pass
            j = min(T, i + self.chunk); d_ = dt[:, i:j]
            log_keep = (d_.unsqueeze(-1) * A).clamp(min=-10)                    # log of how much old state survives each step
            write = (d_ * x[:, i:j]).unsqueeze(-1) * B[:, i:j].unsqueeze(2)     # what each input adds to the state
            if self.chunk == 1:                                                 # the recurrence exactly as written above
                hs = (torch.exp(log_keep[:, 0]) * h + write[:, 0]).unsqueeze(1)
            else:                                                               # the same recurrence for several steps at once:
                P = log_keep.cumsum(1)                                          # h_t = e^P_t * (h_start + sum_{s<=t} e^-P_s * write_s)
                hs = torch.exp(P) * (h.unsqueeze(1) + (torch.exp(-P) * write).cumsum(1))
            ys.append((hs * C[:, i:j].unsqueeze(2)).sum(-1)); h = hs[:, -1]
        y = torch.cat(ys, 1) + x * self.D
        return self.out(y * F.silu(z))

class Block(nn.Module):
    """Pre-norm residual block, so that SSM layers stack the way transformer layers do."""
    def __init__(self, d): super().__init__(); self.norm, self.ssm = nn.LayerNorm(d), SelectiveSSM(d)
    def forward(self, x): return x + self.ssm(self.norm(x))

if __name__ == "__main__":
    import time
    b = Block(128); x = torch.randn(32, 256, 128)
    print(sum(p.numel() for p in b.parameters()), "parameters")
    t = time.time(); y = b(x); y.sum().backward(); print(y.shape, f"{time.time() - t:.2f} s forward + backward")
    slow = Block(128); slow.load_state_dict(b.state_dict()); slow.ssm.chunk = 1
    print("chunked scan equals step-by-step scan:", torch.allclose(b(x), slow(x), atol=1e-4), f"(largest difference {(b(x) - slow(x)).abs().max():.1e})")
    x2 = x.clone(); x2[:, 200:] += 1.0                                          # changing the future must not change the past
    print("causal:", torch.allclose(b(x)[:, :200], b(x2)[:, :200], atol=1e-5))
