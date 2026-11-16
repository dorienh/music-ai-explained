import os; os.makedirs("out", exist_ok=True)
import numpy as np, librosa
rng = np.random.default_rng(0)
def feats(name, dur=60):
    y, sr = librosa.load(f"../data/audio/{name}.wav", sr=22050, duration=dur)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    return np.log(M + 1e-5).T
Xtr = np.concatenate([feats(n) for n in ["nutcracker","brahms","vibeace","sweetwaltz","fishin"]])
Xte = feats("pistachio")
mu = Xtr.mean(0); Xtr -= mu; Xte -= mu
print("train frames", Xtr.shape, "test frames", Xte.shape)
def assign(X, C):
    d = (X**2).sum(1)[:,None] - 2*X@C.T + (C**2).sum(1)[None]
    return d.argmin(1)
def kmeans(X, K, iters=20):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        idx = assign(X, C)
        for k in range(K):
            m = idx==k
            if m.any(): C[k] = X[m].mean(0)
    return C
def rel(X, Xq): return ((X-Xq)**2).sum()/(X**2).sum()
for K in [16, 256, 1024]:
    C = kmeans(Xtr, K)
    print(f"single VQ K={K:5d} bits={np.log2(K):.0f} train err={rel(Xtr,C[assign(Xtr,C)]):.3f} test err={rel(Xte,C[assign(Xte,C)]):.3f}")
res = Xtr.copy(); qtr = np.zeros_like(Xtr); qte = np.zeros_like(Xte)
for s in range(8):
    C = kmeans(res, 16)
    qtr += C[assign(Xtr-qtr, C)]; res = Xtr - qtr
    i = assign(Xte-qte, C); qte += C[i]
    p = np.bincount(i, minlength=16)/len(i); p=p[p>0]
    print(f"RVQ stage {s+1}: bits={4*(s+1):2d} stored={16*(s+1):3d} train err={rel(Xtr,qtr):.3f} test err={rel(Xte,qte):.3f} test entropy={-(p*np.log2(p)).sum():.2f}/4")
