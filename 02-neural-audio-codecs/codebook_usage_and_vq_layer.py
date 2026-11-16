import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, librosa, dac
from encodec import EncodecModel
torch.set_grad_enabled(False)
names = ["nutcracker","brahms","vibeace","sweetwaltz","fishin","pistachio","choice"]
enc = EncodecModel.encodec_model_24khz().eval(); enc.set_target_bandwidth(24.0)
d = dac.DAC.load(dac.utils.download(model_type="44khz")).eval()
E_codes, D_codes = [], []
for n in names:
    y, sr = librosa.load(f"../data/audio/{n}.wav", sr=None, mono=True, duration=60)
    for s in range(0, len(y) - sr*5, sr*20):
        seg = y[s:s+sr*20]
        x = torch.from_numpy(librosa.resample(seg, orig_sr=sr, target_sr=24000))[None, None]
        E_codes.append(torch.cat([c for c,_ in enc.encode(x)], -1)[0])
        x = d.preprocess(torch.from_numpy(seg)[None, None], 44100)
        D_codes.append(d.encode(x)[1][0])
E = torch.cat(E_codes, -1).numpy(); D = torch.cat(D_codes, -1).numpy()
def H(row, K=1024):
    p = np.bincount(row, minlength=K)/len(row); used = (p>0).sum(); p = p[p>0]
    return -(p*np.log2(p)).sum(), used
print("EnCodec frames", E.shape, "DAC frames", D.shape)
for name, C in [("EnCodec", E), ("DAC", D)]:
    for i in range(C.shape[0]):
        h, u = H(C[i]); 
        if i < 9 or i in (15, 31): print(f"{name} cb{i+1:2d}: entropy {h:.2f}/10 bits, perplexity {2**h:6.0f}, entries used {u}/1024")

print("== minimal VQ layer with straight-through, trained")
torch.set_grad_enabled(True); torch.manual_seed(0)
import torch.nn as nn, torch.nn.functional as F
class VQ(nn.Module):
    def __init__(self, K, D, beta=0.25):
        super().__init__(); self.codebook = nn.Embedding(K, D); self.beta = beta
    def forward(self, z):                        # z: [N, D]
        dist = torch.cdist(z, self.codebook.weight)   # [N, K]
        idx = dist.argmin(-1)
        q = self.codebook(idx)
        loss = F.mse_loss(q, z.detach()) + self.beta * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()                 # straight-through
        return q, idx, loss
data = torch.randn(4096, 2) * 0.3 + torch.tensor([[1.,1.],[1.,3.],[3.,1.],[3.,3.]]).repeat(1024,1)
enc_ = nn.Linear(2, 2); dec_ = nn.Linear(2, 2); vq = VQ(4, 2)
opt = torch.optim.Adam([*enc_.parameters(), *dec_.parameters(), *vq.parameters()], 1e-2)
for step in range(2001):
    z = enc_(data); q, idx, l_vq = vq(z); out = dec_(q)
    loss = F.mse_loss(out, data) + l_vq
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(step, "recon", round(F.mse_loss(out, data).item(),4), "entries used", idx.unique().numel(), "enc grad nonzero", bool(enc_.weight.grad.abs().sum()>0))
