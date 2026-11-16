"""Experiments for 'How Text Becomes Music'. Prints every number quoted in the article.
Needs the test clips: python ../data/get_audio.py"""
import os; os.makedirs("out", exist_ok=True)
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, torch, librosa
from transformers import AutoTokenizer, T5EncoderModel, ClapModel, ClapProcessor, logging
logging.set_verbosity_error(); torch.set_grad_enabled(False); np.set_printoptions(precision=2, suppress=True)
A = "../data/audio"; cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=-1))

print("== 1. tokenisation and T5 embeddings (the text encoder inside MusicGen)")
tok = AutoTokenizer.from_pretrained("t5-base"); t5 = T5EncoderModel.from_pretrained("t5-base").eval()
prompt = "sad jazz with brushed drums"
ids = tok(prompt, return_tensors="pt")
print("pieces:", tok.convert_ids_to_tokens(ids.input_ids[0])); print("ids:", ids.input_ids[0].tolist())
out = t5(**ids).last_hidden_state
print("output shape:", tuple(out.shape), "| vocabulary:", tok.vocab_size, "| encoder parameters (M):", round(sum(p.numel() for p in t5.parameters()) / 1e6))
def word_vec(sentence, word):
    e = tok(sentence, return_tensors="pt"); pieces = tok.convert_ids_to_tokens(e.input_ids[0])
    i = [k for k, p in enumerate(pieces) if p.strip("▁") == word][0]
    return t5(**e).last_hidden_state[0, i], t5.get_input_embeddings()(e.input_ids)[0, i]
a_out, a_in = word_vec("a deep bass line with heavy drums", "bass")
b_out, b_in = word_vec("a slow bass groove on an upright", "bass")
c_out, c_in = word_vec("we caught a bass in the lake", "bass")
print("same word 'bass': input embeddings identical:", bool(torch.allclose(a_in, c_in)))
print("after the encoder: music vs music %.2f, music vs fish %.2f" % (cos(a_out, b_out), cos(a_out, c_out)))

print("== 2. attention by hand")
keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]); query = np.array([2.0, 0.5])
scores = keys @ query; w = np.exp(scores) / np.exp(scores).sum()
values = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
print("scores", scores, "weights", w, "output", w @ values)

print("== 3. contrastive loss by hand")
S = np.array([[0.8, 0.1, 0.3], [0.2, 0.7, 0.1], [0.4, 0.2, 0.6]]); temp = 0.2
P = np.exp(S / temp) / np.exp(S / temp).sum(1, keepdims=True)
print("row-wise probabilities\n", P, "\nloss per row", -np.log(np.diag(P)), "mean", -np.log(np.diag(P)).mean())

print("== 4. CLAP: text against audio")
name = "laion/clap-htsat-unfused"
clap = ClapModel.from_pretrained(name).eval(); proc = ClapProcessor.from_pretrained(name)
def audio_emb(waves):
    f = proc(audios=waves, sampling_rate=48000, return_tensors="pt")
    return torch.nn.functional.normalize(clap.get_audio_features(**f), dim=-1)
def text_emb(texts):
    f = proc(text=texts, return_tensors="pt", padding=True)
    return torch.nn.functional.normalize(clap.get_text_features(**f), dim=-1)
clips = {"trumpet": (0, "a solo trumpet"), "brahms": (0, "a string orchestra playing a fast dance"), "nutcracker": (0, "a celesta playing a delicate melody"),
         "choice": (0, "a drum and bass loop"), "fishin": (30, "a folk pop song with a female singer"), "pistachio": (0, "ragtime piano"),
         "vibeace": (0, "a jazz combo with vibraphone, piano and bass"), "libri1": (0, "a person reading a book aloud")}
waves = [librosa.load(f"{A}/{n}.wav", sr=48000, mono=True, offset=o, duration=10)[0] for n, (o, _) in clips.items()]
AE = audio_emb(waves); TE = text_emb([c for _, c in clips.values()]); M = (TE @ AE.T).numpy()
print("embedding size:", AE.shape[1]); print("rows = captions, columns = clips", list(clips)); print(M)
print("caption -> best clip correct:", int((M.argmax(1) == np.arange(len(clips))).sum()), "of", len(clips), "| clip -> best caption correct:", int((M.argmax(0) == np.arange(len(clips))).sum()))
print("mean matching %.2f, mean non-matching %.2f" % (np.diag(M).mean(), (M.sum() - np.trace(M)) / (M.size - len(clips))))
json.dump({"clips": list(clips), "captions": [c for _, c in clips.values()], "sim": M.tolist()}, open("clap_matrix.json", "w"))

print("== 5. where CLAP is blind")
pairs = [("jazz with drums", "jazz without drums"), ("piano, then drums", "drums, then piano"), ("a slow drum loop", "a fast drum loop"),
         ("a song in a major key", "a song in a minor key"), ("sad jazz with brushed drums", "happy jazz with brushed drums"), ("sad jazz with brushed drums", "melancholic jazz with soft brush drumming"),
         ("sad jazz with brushed drums", "heavy metal with distorted guitars")]
for a, b in pairs:
    e = text_emb([a, b]); print(f"text-text {float(e[0] @ e[1]):.2f}   '{a}' | '{b}'")
drum = waves[list(clips).index("choice")]
E2 = text_emb(["music with drums", "music without drums", "no drums"]); d = audio_emb([drum])
print("drum loop vs 'music with drums' %.2f, 'music without drums' %.2f, 'no drums' %.2f" % tuple((E2 @ d.T).squeeze().tolist()))
jazz = waves[list(clips).index("vibeace")]; trumpet = waves[list(clips).index("trumpet")]; j = audio_emb([trumpet])
print("solo trumpet (no drums) vs the same three: %.2f %.2f %.2f" % tuple((E2 @ j.T).squeeze().tolist()))
slow, fast = librosa.effects.time_stretch(drum, rate=0.6)[:480000], librosa.effects.time_stretch(np.tile(drum, 2), rate=1.6)[:480000]
E3 = text_emb(["a slow drum loop", "a fast drum loop"]); sf_ = audio_emb([slow, fast])
print("tempo: rows = slowed clip, sped-up clip; cols = 'slow', 'fast'\n", (sf_ @ E3.T).numpy())
