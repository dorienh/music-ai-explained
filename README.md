# music-ai-explained

Code for my articles on audio and music AI.

Runnable code behind the explainers I write on how audio and music AI works. Every number, table and figure in an article comes from a script in this repository, so you can rerun it, change it and check it.

By Dorien Herremans. Articles: [Music & AI on Medium](https://medium.com/@dorien-herremans/list/music-ai-3fb88d4826dd). Research papers: [dorienherremans.com/biblio](https://dorienherremans.com/biblio).

## Articles

| Folder | Article | What the code does |
|---|---|---|
| `01-what-is-an-audio-codec/` | What Is an Audio Codec? From MP3 to the Tokens Behind AI Music `[link]` | Aliasing, bit depth and noise, lossless prediction, hearing thresholds and masking, MP3 bandwidth and pre-echo, with two figures |
| `02-neural-audio-codecs/` | Neural Audio Codecs Up Close `[link]` | One codebook against residual vector quantization on real music, codebook usage of EnCodec and DAC, a minimal trainable VQ layer |
| `03-comparing-codecs/` | I Ran Five Neural Audio Codecs on the Same Music `[link]` | EnCodec, DAC, SNAC, Mimi, Opus and MP3 on the same clips, scored with SI-SDR, mel distance, PESQ and STOI |
| `04-autoregressive-diffusion-flow-matching/` | Autoregressive, Diffusion, Flow Matching `[link]` | Toy diffusion against flow matching, MusicGen and AudioLDM 2 timings |
| `05-how-text-becomes-music/` | How Text Becomes Music `[link]` | T5 tokens and embeddings, attention and contrastive loss by hand, CLAP on real recordings and its blind spots, guidance against prompt adherence |
| `06-generating-full-songs/` | How AI Generates a Full Song With Vocals `[link]` | Sequence lengths, lyrics laid out on a time grid (sentence- and word-level), self-similarity of real songs against chained MusicGen windows, words lost when a mix is coded as one token stream |
| `07-symbolic-vs-audio/` | Notes or Sound? Symbolic and Audio Music Generation Compared `[link]` | Four notes written six ways, six tokenizations of the Bach chorales against audio codecs, a trumpet turned into notes and back, a small transformer trained on 371 chorales, text2midi against MusicGen on one prompt |
| `08-open-vs-closed/` | Open or Closed? What It Really Takes to Run an AI Music Model Yourself `[link]` | Memory from parameter counts, measured memory and speed of MusicGen-small and -medium, weights rounded to fewer bits, cost per song |
| `09-new-backbones/` | Does Music AI Need a New Engine? Transformers Against State Space Models `[link]` | A selective state space block (the core of Mamba) in plain PyTorch, timed against attention from 512 to 65,536 frames, and trained against a transformer on the Bach chorales |

## Notebooks

Each article has a notebook that walks through its worked examples, with figures and audio players. They run in the browser on a free Colab CPU.

| Notebook | Open |
|---|---|
| What is an audio codec? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/01-what-is-an-audio-codec/notebook.ipynb) |
| Codebooks and residual vector quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/02-neural-audio-codecs/notebook.ipynb) |
| Running and comparing codecs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/03-comparing-codecs/notebook.ipynb) |
| Autoregressive, diffusion, flow matching | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/04-autoregressive-diffusion-flow-matching/notebook.ipynb) |
| How text becomes music | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/05-how-text-becomes-music/notebook.ipynb) |
| Generating full songs with vocals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/06-generating-full-songs/notebook.ipynb) |
| Notes or sound? Symbolic and audio generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/07-symbolic-vs-audio/notebook.ipynb) |
| Open or closed? Running a music model yourself | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/08-open-vs-closed/notebook.ipynb) |
| Transformers against state space models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorienh/music-ai-explained/blob/main/09-new-backbones/notebook.ipynb) |

The notebooks are saved with their printed results and figures. Audio players are left out of the saved copy; run a cell to listen.

The scripts next to each notebook produce the exact numbers, tables and figures quoted in the articles, including the slower benchmarks.

## Running the scripts

```bash
pip install -r requirements.txt
python data/get_audio.py            # downloads freely licensed test clips into data/audio/

cd 01-what-is-an-audio-codec
python experiments.py               # prints the article's numbers, writes figures to out/
```

Each folder runs on its own from inside that folder and writes to `out/`. The `.json` files next to the scripts hold the numbers from the run quoted in the article.

Notes:

- `ffmpeg` with libmp3lame and libopus is needed for the MP3 and Opus comparisons.
- Everything runs on a laptop CPU except `04-.../ldm_timing.py`, which wants a GPU (`python ldm_timing.py mps` or `cuda`).
- Model weights download on first use (a few GB in total for folders 03 to 07; `08-.../measure_local.py` adds 8 GB for MusicGen-medium).
- `07-.../text_to_notes_vs_text_to_sound.py` needs a clone of [text2midi](https://github.com/AMAAI-Lab/text2midi) next to this repository; the script's first lines say how.
- The published numbers were produced on an Apple silicon Mac with Python 3.12, torch 2.3, transformers 4.47 and diffusers 0.31. Timings will differ on your machine; the other results should match closely.

## Learn more

For the full picture of music audio generation, see our 132-page tutorial: Modrzejewski, Kurkowska, Herremans (2026), [A Tutorial on Generative Models for Music Audio](https://arxiv.org/abs/2610.38252).

## Figures

`assets/` holds the figures and table images used in the published articles.

## Licence

Code: MIT, see `LICENSE`.

## Test audio

The clips are the example recordings distributed with [librosa](https://librosa.org/doc/latest/recordings.html). They are not stored in this repository. See that page for each recording's licence.

## Listening clips

Each article folder has an `audio/` folder with the clips the article refers to: codec outputs, generated music and the like. Clips derived from the librosa recordings keep the licence of the recording they come from. Bach chorales come from the [music21](https://www.music21.org/) corpus. `07-symbolic-vs-audio/audio/text2midi_pop.wav` was rendered with FluidSynth and its bundled Vintage Dreams Waves soundfont (freeware, © Ian Wilson).
