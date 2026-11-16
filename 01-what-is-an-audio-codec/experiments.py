import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import os, subprocess, numpy as np, librosa, soundfile as sf
from scipy.signal import correlate, stft, istft
from scipy.fft import dct
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A, OUT = "../data/audio", "out"
BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "Helvetica Neue",
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2})

print("== 1 aliasing")
sr = 44100; n = np.arange(sr); x = np.sin(2*np.pi*30000*n/sr)
print("30 kHz tone sampled at 44.1 kHz shows up at", np.abs(np.fft.rfft(x)).argmax(), "Hz")
x = np.sin(2*np.pi*25000*n/sr); print("25 kHz ->", np.abs(np.fft.rfft(x)).argmax(), "Hz")

print("== 2 bit depth SNR, full-scale 997 Hz sine")
x = np.sin(2*np.pi*997*np.arange(sr*2)/sr)
for b in [4, 8, 12, 16]:
    q = np.round(x * (2**(b-1)-1)) / (2**(b-1)-1)
    print(f"{b:2d} bits: levels {2**b:6d}  measured SNR {10*np.log10((x**2).sum()/((x-q)**2).sum()):6.2f} dB   6.02b+1.76 = {6.02*b+1.76:.2f}")

print("== 3 lossless: entropy of samples vs prediction residuals (bits/sample), and FLAC")
def H(v):
    _, c = np.unique(v, return_counts=True); p = c/c.sum(); return -(p*np.log2(p)).sum()
for name in ["libri1", "choice", "brahms", "fishin", "trumpet"]:
    y, s = librosa.load(f"{A}/{name}.wav", sr=None, mono=True, duration=30)
    v = np.round(y*32767).astype(np.int64)
    d1 = v[1:] - v[:-1]; d2 = v[2:] - (2*v[1:-1] - v[:-2])
    sf.write(f"{OUT}/_t.wav", v.astype(np.int16), s, subtype="PCM_16")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_t.wav","-compression_level","8",f"{OUT}/_t.flac"], check=True)
    ratio = os.path.getsize(f"{OUT}/_t.flac")/os.path.getsize(f"{OUT}/_t.wav")
    print(f"{name:8s} sr={s} raw {H(v):5.2f}  predict-previous {H(d1):5.2f}  predict-line {H(d2):5.2f}  FLAC size {100*ratio:4.1f}% = {16*ratio:4.1f} bits/sample")
v = np.array([100, 104, 109, 113, 116]); print("toy", v, "prev-residual", v[1:]-v[:-1], "line-residual", v[2:]-(2*v[1:-1]-v[:-2]))

print("== 4 threshold in quiet (Terhardt) and Bark")
ath = lambda f: 3.64*(f/1000)**-0.8 - 6.5*np.exp(-0.6*(f/1000-3.3)**2) + 1e-3*(f/1000)**4
bark = lambda f: 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)
for f in [50, 100, 500, 1000, 3300, 10000, 15000, 18000]:
    print(f"{f:6d} Hz: threshold {ath(f):6.1f} dB SPL, {bark(f):5.2f} Bark")
L, fm = 80.0, 1000.0; zm = bark(fm)
def mask(f):
    dz = bark(f) - zm
    return L - 14.5 - zm - np.where(dz < 0, 25*(-dz), 10*dz)
for f in [700, 850, 1000, 1170, 1370, 2000, 4000]:
    print(f"masker 80 dB @1 kHz: threshold at {f} Hz = {float(mask(f)):5.1f} dB (quiet: {ath(f):4.1f}), dz={bark(f)-zm:+.2f}")
print("bits needed in masker band: (80 - %.1f)/6.02 = %.1f" % (mask(1000.), (80-mask(1000.))/6.02))
f = np.geomspace(30, 18000, 800)
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(f, ath(f), color=INK2, lw=2, label="Threshold in quiet")
m = np.maximum(mask(f), ath(f)); sel = mask(f) > ath(f)
ax.plot(f[sel], m[sel], color=BLUE, lw=2, label="Threshold with the tone playing")
ax.fill_between(f[sel], ath(f)[sel], m[sel], color=BLUE, alpha=0.12, lw=0)
ax.vlines(1000, ath(1000), 80, color=ORANGE, lw=3); ax.plot([1000], [80], "o", color=ORANGE, ms=8)
ax.annotate("80 dB tone at 1 kHz", (1000, 80), xytext=(-10, 0), textcoords="offset points", color=INK, ha="right", va="center")
ax.annotate("Threshold while\nthe tone plays", (1250, 50), color=BLUE, fontsize=10)
ax.annotate("Hidden: anything in\nthe shaded area", (1700, 30), color=INK, fontsize=10)
ax.annotate("Threshold in quiet", (60, ath(60)+3), color=INK2, fontsize=10)
ax.set_xscale("log"); ax.set_xlim(30, 18000); ax.set_ylim(-10, 90)
ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k"])
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Sound level (dB SPL)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_masking.png", dpi=200); plt.close(fig)

print("== 5 energy compaction: fraction of DCT coefficients holding 99% of energy; SNR keeping top 10%")
for name in ["choice", "brahms", "fishin"]:
    y, s = librosa.load(f"{A}/{name}.wav", sr=None, mono=True, duration=20)
    fr = y[:len(y)//2048*2048].reshape(-1, 2048); C = dct(fr, norm="ortho", axis=1)
    e = np.sort(C**2, axis=1)[:, ::-1]; cum = np.cumsum(e, 1)/e.sum(1, keepdims=True)
    n99 = (cum < 0.99).sum(1) + 1
    k = 205; thr = np.sort(np.abs(C), 1)[:, -k][:, None]; Ck = np.where(np.abs(C) >= thr, C, 0)
    tk = (np.sort(fr**2, 1)[:, ::-1][:, :k].sum()/ (fr**2).sum())
    print(f"{name:8s} median coeffs for 99% energy: {np.median(n99):5.0f}/2048 ({100*np.median(n99)/2048:.1f}%)  keep top 10% coeffs -> SNR {10*np.log10((C**2).sum()/((C-Ck)**2).sum()):.1f} dB ; top 10% of raw samples hold {100*tk:.0f}% energy")

print("== 6 same noise energy, white vs shaped under the spectrum")
y, s = librosa.load(f"{A}/fishin.wav", sr=None, mono=True, offset=30, duration=10)
rng = np.random.default_rng(0); target_snr = 20
w = rng.standard_normal(len(y)); w *= np.sqrt((y**2).sum()/10**(target_snr/10)/(w**2).sum())
_, _, Y = stft(y, s, nperseg=2048); _, _, W = stft(rng.standard_normal(len(y)), s, nperseg=2048)
_, shaped = istft(W*np.abs(Y), s, nperseg=2048); shaped = shaped[:len(y)]
shaped *= np.sqrt((y**2).sum()/10**(target_snr/10)/(shaped**2).sum())
snr = lambda n: 10*np.log10((y**2).sum()/(n**2).sum())
print("SNR white %.2f dB, shaped %.2f dB" % (snr(w), snr(shaped)))
sf.write(f"{OUT}/noise_original.wav", y, s); sf.write(f"{OUT}/noise_white_20dB.wav", y+w, s); sf.write(f"{OUT}/noise_shaped_20dB.wav", y+shaped, s)

print("== 7 MP3 bitrates: size, bandwidth")
def bandwidth(z, s):
    P = np.abs(librosa.stft(z, n_fft=4096))**2; p = 10*np.log10(P.mean(1)+1e-20); fr = np.fft.rfftfreq(4096, 1/s)
    return fr[np.where(p > p.max()-70)[0].max()]
y, s = librosa.load(f"{A}/choice.wav", sr=None, mono=True, duration=20)
sf.write(f"{OUT}/_in.wav", y, s, subtype="PCM_16"); wavsize = os.path.getsize(f"{OUT}/_in.wav")
print(f"WAV mono 16-bit: {wavsize} bytes, bandwidth {bandwidth(y,s):.0f} Hz")
for kb in [320, 128, 64, 32]:
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_in.wav","-b:a",f"{kb}k",f"{OUT}/_t.mp3"], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_t.mp3",f"{OUT}/_o.wav"], check=True)
    z, zs = librosa.load(f"{OUT}/_o.wav", sr=None, mono=True)
    print(f"MP3 {kb:3d}k: {os.path.getsize(f'{OUT}/_t.mp3'):7d} bytes ({wavsize/os.path.getsize(f'{OUT}/_t.mp3'):5.1f}x smaller), decoded sr {zs}, bandwidth {bandwidth(z,zs):6.0f} Hz")

print("== 8 pre-echo on a synthetic click")
s = 44100; t = np.arange(int(0.25*s))/s
burst = rng.standard_normal(len(t))*np.exp(-t/0.02)*0.8
x = np.concatenate([np.zeros(int(0.5*s)), burst, np.zeros(int(0.25*s))]); onset = int(0.5*s)
sf.write(f"{OUT}/click.wav", x, s, subtype="PCM_16")
res = {}
for label, args, ext in [("MP3 64k", ["-c:a","libmp3lame","-b:a","64k"], "mp3"), ("MP3 128k", ["-c:a","libmp3lame","-b:a","128k"], "mp3"), ("Opus 64k", ["-c:a","libopus","-b:a","64k"], "opus")]:
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/click.wav",*args,f"{OUT}/_c.{ext}"], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_c.{ext}","-ar","44100",f"{OUT}/_c.wav"], check=True)
    z, _ = sf.read(f"{OUT}/_c.wav"); c = correlate(z, x, mode="full"); lag = c.argmax() - (len(x)-1)
    z = z[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), z]); z = np.pad(z, (0, max(0, len(x)-len(z))))[:len(x)]; res[label] = z
    pk = np.abs(x).max()
    for ms in [5, 20]:
        seg = z[onset-int(ms/1000*s):onset]
        print(f"{label}: lag {lag}, RMS in the {ms} ms before the click = {20*np.log10(np.sqrt((seg**2).mean())/pk+1e-12):6.1f} dB re click peak; ", end="")
    pre = np.where(np.abs(z[:onset]) > pk*10**(-60/20))[0]
    print(f"first sample above -60 dB: {(onset-pre.min())/s*1000 if len(pre) else 0:.1f} ms before onset")
fig, axs = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, sharey=True)
tt = (np.arange(len(x))-onset)/s*1000; w_ = slice(onset-int(0.02*s), onset+int(0.008*s))
for ax, sig, lab, col in [(axs[0], x, "Original: silence, then a click", INK2), (axs[1], res["MP3 64k"], "After MP3 at 64 kbit/s: noise arrives before the click", BLUE)]:
    ax.plot(tt[w_], sig[w_], color=col, lw=1); ax.axvline(0, color=ORANGE, lw=1.5); ax.set_title(lab, loc="left", fontsize=11, color=INK); ax.set_ylabel("Amplitude")
axs[1].set_xlabel("Time relative to the click (ms)"); axs[1].set_ylim(-0.3, 0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_preecho.png", dpi=200); plt.close(fig)
