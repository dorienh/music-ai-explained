import librosa, soundfile as sf, sys, os
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(out, exist_ok=True)
for k in ["libri1","libri2","nutcracker","choice","fishin","brahms","trumpet","vibeace","sweetwaltz","pistachio"]:
    try:
        p = librosa.example(k, hq=True)
        y, sr = librosa.load(p, sr=None, mono=True)
        sf.write(os.path.join(out, k + ".wav"), y, sr)
        print(k, sr, round(len(y)/sr,1), "s")
    except Exception as e:
        print(k, "FAIL", repr(e)[:200])
