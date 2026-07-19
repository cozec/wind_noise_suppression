# Wind Noise Suppression Demo

End-to-end demo: mix clean speech with **real recorded wind noise**, then
suppress the wind with classical DSP and neural methods, evaluated with
SI-SDR / PESQ / STOI.

## How it works

1. **Clean speech** — bundled librosa LibriSpeech example (`libri1`), 8 s @ 16 kHz.
2. **Wind noise** — real mobile-phone recordings (strong wind) from the
   [Zenodo Wind Noise Dataset](https://zenodo.org/records/6687982).
   Synthetic wind via
   [SC-Wind-Noise-Generator](https://github.com/audiolabs/SC-Wind-Noise-Generator)
   was used in an earlier iteration (still cloned in `data/`).
3. **Mix** at 0 dB SNR.
4. **Suppress** with four methods:
   - High-pass filter (150 Hz Butterworth)
   - Spectral gating ([noisereduce](https://github.com/timsainb/noisereduce), non-stationary)
   - [Meta Denoiser](https://github.com/facebookresearch/denoiser) (DNS64, causal, 16 kHz)
   - [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (48 kHz, run in a
     separate py3.11 venv — its Rust extension has no py3.14 wheel)
5. **Evaluate** with SI-SDR, PESQ-WB, and STOI against the clean reference.

See [summary.md](summary.md) for results — TL;DR: on real wind at 0 dB SNR the
neural methods win decisively (Meta Denoiser: +10 dB SI-SDR), while classical
methods that worked on synthetic wind barely help or hurt.

## Setup

```bash
# main env (python 3.14 ok)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib soundfile librosa noisereduce spectrum sounddevice \
            torch torchaudio denoiser pesq pystoi

# DeepFilterNet env (needs python <= 3.11 and torch/torchaudio 2.1.2)
python3.11 -m venv .venv-dfn
.venv-dfn/bin/pip install deepfilternet "torch==2.1.2" "torchaudio==2.1.2" soundfile

# data
curl -L -o data/wind_noise_dataset.zip \
  "https://zenodo.org/records/6687982/files/wind_noise_dataset.zip?download=1"
(cd data && unzip wind_noise_dataset.zip)
```

## Run

```bash
source .venv/bin/activate
python src/wind_noise_demo.py          # main demo: metrics + wavs + spectrograms
cd src && python plot_examples.py      # 2-example waveform+spectrogram comparison grids
```

## Listen

Sample audio from Example 1 (libri1 + strong wind, 0 dB SNR) — click a link,
then use the player on the GitHub file page:

| Audio | SI-SDR (dB) | PESQ-WB | STOI |
|---|---:|---:|---:|
| [Clean speech](results/clean.wav) | — | — | — |
| [Wind only](results/wind_only.wav) | — | — | — |
| [Noisy input (speech + wind)](results/noisy.wav) | -0.18 | 1.05 | 0.846 |
| [High-pass 150 Hz](results/enhanced_highpass.wav) | -2.58 | 1.25 | 0.840 |
| [Spectral gating](results/enhanced_spectral_gate.wav) | -0.64 | 1.16 | 0.827 |
| [Meta Denoiser (DNS64)](results/enhanced_meta_denoiser.wav) | 10.00 | 1.63 | 0.943 |
| [DeepFilterNet](results/enhanced_deepfilternet.wav) | 5.02 | 1.41 | 0.873 |

## Outputs

- `results/*.wav` — clean, wind-only, noisy, and enhanced audio (listen to compare)
- `results/metrics.csv` — SI-SDR / PESQ / STOI per method
- `plots/spectrograms.png` — clean / noisy / enhanced spectrograms
- `plots/example1_strong_wind.png`, `plots/example2_normal_wind.png` —
  per-algorithm waveform + spectrogram grids for two speech+wind mixtures
  (libri1 + strong wind, libri2 + normal wind), metrics in each row title

![Example 1: strong wind](plots/example1_strong_wind.png)

## Datasets

- [Wind Noise Dataset (Zenodo)](https://zenodo.org/records/6687982) — used here:
  478 phone-recorded (normal/strong wind) + 100 generated wind clips, 16 kHz mono
- [Wind Noise Database (RWTH Aachen IKS)](https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/wind-noise-database)
  — lab + outdoor wind recordings (manual download form)
