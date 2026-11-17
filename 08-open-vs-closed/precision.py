"""How few bits per weight can a music model live with?

A model's memory is its parameter count times the bytes per parameter, so storing each weight in fewer bits shrinks it.
This script rounds the weights of MusicGen-small's language model to 16, 8, 6, 5, 4 and 3 bits and measures, on real music:
  loss        how surprised the model is by the real next token (nats per token; lower is better)
  agreement   how often its top guess for the next token is the same as the full-precision model's

The rounding is the simplest possible: every row of every weight matrix is scaled to its largest value and rounded to
evenly spaced levels. Real tools (GGUF, bitsandbytes, GPTQ) are smarter, so read these numbers as a floor, not as the state of the art.
The arithmetic still runs in 32-bit floats here, so this measures quality, not speed.

Run:  python precision.py       (needs the test clips: python ../data/get_audio.py; about 2 minutes on a laptop CPU)
"""
import copy, json, os
import numpy as np, torch, librosa
from transformers import AutoProcessor, MusicgenForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__)); AUDIO = os.path.join(HERE, "..", "data", "audio")
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
CLIPS = {"pistachio": "ragtime piano", "vibeace": "a jazz combo with vibraphone, piano and bass", "choice": "a drum and bass loop",
         "brahms": "an orchestra playing a fast dance", "nutcracker": "a celesta playing a delicate melody", "sweetwaltz": "a slow waltz"}
PAD, K = 2048, 4

@torch.no_grad()
def prepare(name, text):
    """A real clip -> its codec tokens laid out the way MusicGen reads them (codebook k delayed by k steps), plus the encoded text."""
    y, _ = librosa.load(os.path.join(AUDIO, name + ".wav"), sr=32000, mono=True, duration=10)
    a = proc(audio=[y], sampling_rate=32000, return_tensors="pt", padding=True)
    codes = model.audio_encoder.encode(a["input_values"], a["padding_mask"]).audio_codes[0, 0]          # [4, 500]
    T = codes.shape[1]; X = torch.full((K, T + K - 1), PAD)
    for k in range(K): X[k, k:k + T] = codes[k]
    inp = torch.cat([torch.full((K, 1), PAD), X[:, :-1]], 1)                                               # the model sees everything before
    t = proc(text=[text], return_tensors="pt", padding=True)
    enc = model.enc_to_dec_proj(model.text_encoder(**t).last_hidden_state)
    return inp, X, enc, t["attention_mask"]
data = [prepare(n, t) for n, t in CLIPS.items()]

@torch.no_grad()
def evaluate(decoder):
    losses, preds = [], []
    for inp, X, enc, mask in data:
        logits = decoder(input_ids=inp, encoder_hidden_states=enc, encoder_attention_mask=mask).logits.reshape(K, X.shape[1], -1)
        valid = X != PAD
        losses.append(torch.nn.functional.cross_entropy(logits[valid], X[valid]).item()); preds.append(logits[valid].argmax(-1))
    return float(np.mean(losses)), torch.cat(preds)

def rounded(decoder, bits):
    d = copy.deepcopy(decoder)
    for m in d.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            if bits == 16: m.weight.data = w.half().float(); continue
            levels = 2 ** (bits - 1) - 1                                         # e.g. 127 steps either side of zero for 8 bits
            scale = w.abs().amax(dim=1, keepdim=True) / levels
            m.weight.data = torch.round(w / scale).clamp(-levels, levels) * scale
    return d

n_lin = sum(m.weight.numel() for m in model.decoder.modules() if isinstance(m, torch.nn.Linear))
n_all = sum(p.numel() for p in model.decoder.parameters())
print(f"MusicGen-small language model: {n_all / 1e6:.0f} M parameters, {n_lin / 1e6:.0f} M of them in the weight matrices that get rounded")
base_loss, base_pred = evaluate(model.decoder)
print(f"evaluated on {len(data)} real clips of 10 s = {len(base_pred):,} tokens; a uniform guess would score {np.log(2048):.2f}\n")
print(f"{'bits per weight':>16s} {'size of the matrices':>22s} {'loss':>7s} {'change':>8s} {'same top guess':>15s}")
res = {32: {"mb": n_lin * 4 / 1e6, "loss": base_loss, "agree": 1.0}}
print(f"{32:>16d} {n_lin * 4 / 1e6:>19.0f} MB {base_loss:>7.3f} {'':>8s} {'100.0%':>15s}")
for bits in (16, 8, 6, 5, 4, 3):
    loss, pred = evaluate(rounded(model.decoder, bits)); agree = float((pred == base_pred).float().mean())
    res[bits] = {"mb": n_lin * bits / 8 / 1e6, "loss": loss, "agree": agree}
    print(f"{bits:>16d} {n_lin * bits / 8 / 1e6:>19.0f} MB {loss:>7.3f} {loss - base_loss:>+8.3f} {100 * agree:>14.1f}%")
json.dump({"params_lm": n_all, "params_rounded": n_lin, "tokens": len(base_pred), "results": res}, open(os.path.join(HERE, "precision.json"), "w"), indent=1)

# ---------- listen: the same prompt and seed at 32, 8 and 4 bits ----------
import soundfile as sf
adir = os.path.join(HERE, "audio"); os.makedirs(adir, exist_ok=True)
x = proc(text=["upbeat folk pop song with acoustic guitar, bass and drums"], return_tensors="pt"); original = model.decoder
for bits in (32, 8, 4):
    model.decoder = original if bits == 32 else rounded(original, bits); torch.manual_seed(0)
    with torch.no_grad(): y = model.generate(**x, do_sample=True, guidance_scale=3, max_new_tokens=400)[0, 0].numpy()
    sf.write(os.path.join(adir, f"musicgen-small_{bits}bit.wav"), y, 32000); print("wrote", f"musicgen-small_{bits}bit.wav")
model.decoder = original
