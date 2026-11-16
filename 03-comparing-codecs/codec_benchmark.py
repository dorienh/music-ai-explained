import os; os.makedirs("out", exist_ok=True)
import os, time, subprocess, warnings, json, sys
warnings.filterwarnings("ignore")
import numpy as np, torch, librosa, soundfile as sf
from scipy.signal import correlate
from pesq import pesq
from pystoi import stoi
torch.set_grad_enabled(False); torch.set_num_threads(8)
A = "../data/audio"; OUT = "out"; os.makedirs(OUT, exist_ok=True)

def load(name, start, dur):
    y, sr = librosa.load(f"{A}/{name}.wav", sr=None, mono=True, offset=start, duration=dur)
    return y.astype(np.float32), sr
CLIPS = {"speech": load("libri1", 0, 14.8), "drums": load("choice", 0, 20), "strings": load("brahms", 0, 20), "vocals": load("fishin", 30, 20)}
rs = lambda y, a, b: y if a == b else librosa.resample(y, orig_sr=a, target_sr=b, res_type="soxr_hq")

def align(ref, est, maxlag):
    n = min(len(ref), len(est), 16000*5)
    c = correlate(est[:n], ref[:n], mode="full"); lags = np.arange(-n+1, n)
    m = np.abs(lags) <= maxlag; lag = lags[m][np.abs(c[m]).argmax()]
    if lag > 0: est = est[lag:]
    elif lag < 0: est = np.concatenate([np.zeros(-lag, dtype=est.dtype), est])
    n = min(len(ref), len(est)); return ref[:n], est[:n], lag
def si_sdr(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    t = (est @ ref) / (ref @ ref) * ref
    return 10*np.log10((t @ t) / ((est - t) @ (est - t)))
def mel_dist(ref, est, sr):
    f = lambda y: np.log10(np.maximum(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80, power=1.0), 1e-5))
    return float(np.abs(f(ref) - f(est)).mean())

# ---------- codecs: each returns (decoded audio, sr, info dict) ----------
M = {}
def encodec24(y, sr, bw):
    from encodec import EncodecModel
    m = M.setdefault("enc24", EncodecModel.encodec_model_24khz().eval()); m.set_target_bandwidth(bw)
    x = torch.from_numpy(rs(y, sr, 24000))[None, None]
    frames = m.encode(x); codes = torch.cat([c for c, _ in frames], -1)
    return m.decode(frames)[0, 0].numpy(), 24000, codes
def encodec32(y, sr):
    from transformers import EncodecModel, AutoProcessor
    m = M.setdefault("enc32", EncodecModel.from_pretrained("facebook/encodec_32khz").eval())
    p = M.setdefault("enc32p", AutoProcessor.from_pretrained("facebook/encodec_32khz"))
    inp = p(raw_audio=rs(y, sr, 32000), sampling_rate=32000, return_tensors="pt")
    e = m.encode(inp["input_values"], inp["padding_mask"])
    out = m.decode(e.audio_codes, e.audio_scales, inp["padding_mask"])[0]
    return out[0, 0].numpy(), 32000, e.audio_codes[0]
def dac44(y, sr, nq=None):
    import dac
    m = M.setdefault("dac44", dac.DAC.load(dac.utils.download(model_type="44khz")).eval())
    x = m.preprocess(torch.from_numpy(rs(y, sr, 44100))[None, None], 44100)
    z, codes, *_ = m.encode(x, n_quantizers=nq)
    if nq: codes = codes[:, :nq]
    return m.decode(z)[0, 0].numpy(), 44100, codes
def snac_(y, sr, name, msr):
    from snac import SNAC
    m = M.setdefault(name, SNAC.from_pretrained(f"hubertsiuzdak/{name}").eval())
    x = torch.from_numpy(rs(y, sr, msr))[None, None]
    codes = m.encode(x)
    return m.decode(codes)[0, 0].numpy(), msr, codes
def mimi(y, sr, nq=8):
    from transformers import MimiModel
    m = M.setdefault("mimi", MimiModel.from_pretrained("kyutai/mimi").eval())
    x = torch.from_numpy(rs(y, sr, 24000))[None, None]
    e = m.encode(x, num_quantizers=nq)
    return m.decode(e.audio_codes)[0][0, 0].numpy(), 24000, e.audio_codes
def ff(y, sr, codec, kbps, ext):
    osr = 48000 if codec == "libopus" else 44100
    sf.write(f"{OUT}/_in.wav", rs(y, sr, osr), osr)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_in.wav","-c:a",codec,"-b:a",f"{kbps}k",f"{OUT}/_t.{ext}"], check=True)
    size = os.path.getsize(f"{OUT}/_t.{ext}")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{OUT}/_t.{ext}",f"{OUT}/_out.wav"], check=True)
    o, osr2 = sf.read(f"{OUT}/_out.wav", dtype="float32")
    return o, osr2, size*8/(len(y)/sr)

SYSTEMS = [
 ("EnCodec 24k, 2 cb",  lambda y,sr: encodec24(y,sr,1.5)),
 ("EnCodec 24k, 8 cb",  lambda y,sr: encodec24(y,sr,6.0)),
 ("EnCodec 24k, 32 cb", lambda y,sr: encodec24(y,sr,24.0)),
 ("EnCodec 32k (MusicGen)", encodec32),
 ("DAC 44k, 3 cb",      lambda y,sr: dac44(y,sr,3)),
 ("DAC 44k, 9 cb",      lambda y,sr: dac44(y,sr,None)),
 ("SNAC 44k",           lambda y,sr: snac_(y,sr,"snac_44khz",44100)),
 ("SNAC 24k (speech)",  lambda y,sr: snac_(y,sr,"snac_24khz",24000)),
 ("Mimi, 8 cb",         lambda y,sr: mimi(y,sr,8)),
 ("Opus 6k",            lambda y,sr: ff(y,sr,"libopus",6,"opus")),
 ("Opus 12k",           lambda y,sr: ff(y,sr,"libopus",12,"opus")),
 ("MP3 32k",            lambda y,sr: ff(y,sr,"libmp3lame",32,"mp3")),
 ("MP3 128k",           lambda y,sr: ff(y,sr,"libmp3lame",128,"mp3")),
]
rows = []
for name, fn in SYSTEMS:
    for cname, (y, sr) in CLIPS.items():
        try:
            fn(y[:sr], sr)  # warm-up / load
            t = time.time(); est, esr, info = fn(y, sr); dt = time.time() - t
        except Exception as e:
            print("FAIL", name, cname, repr(e)[:300]); continue
        dur = len(y)/sr
        if isinstance(info, float): kbps, shape = info/1000, "-"
        elif isinstance(info, list):
            shape = [tuple(c.shape) for c in info]
            bits = {"SNAC 44k":12, "SNAC 24k (speech)":12}[name]
            kbps = sum(c.shape[-1] for c in info)*bits/dur/1000
        else:
            shape = tuple(info.shape); nq, T = info.shape[-2], info.shape[-1]
            bits = 11 if ("Mimi" in name or "32k" in name) else 10
            kbps = nq*T*bits/dur/1000
        r16, e16, lag = align(rs(y, sr, 16000), rs(est, esr, 16000), 1600)
        row = dict(system=name, clip=cname, kbps=round(kbps,2), shape=str(shape), lag16=int(lag),
                   si_sdr=round(float(si_sdr(r16,e16)),2), mel16=round(mel_dist(r16,e16,16000),3), rtf=round(dt/dur,3))
        if sr >= 44100:
            lag44 = int(round(lag*44100/16000)); e44 = rs(est, esr, 44100); r44 = y
            if lag44 > 0: e44 = e44[lag44:]
            elif lag44 < 0: e44 = np.concatenate([np.zeros(-lag44, dtype=np.float32), e44])
            n = min(len(r44), len(e44)); row["mel44"] = round(mel_dist(r44[:n], e44[:n], 44100),3)
        if cname == "speech":
            row["pesq"] = round(pesq(16000, r16, e16, "wb"),2); row["stoi"] = round(float(stoi(r16, e16, 16000)),3)
        rows.append(row); print(json.dumps(row), flush=True)
        if cname in ("speech","vocals"):
            tag = name.replace(" ","_").replace(",","").replace("(","").replace(")","")
            sf.write(f"{OUT}/{cname}__{tag}.wav", est[:int(esr*10)], esr)
for cname in ("speech","vocals"):
    y, sr = CLIPS[cname]; sf.write(f"{OUT}/{cname}__original.wav", y[:sr*10], sr)
json.dump(rows, open(f"{OUT}/rows.json","w"), indent=1)
