"""Wind noise suppression demo.

Pipeline:
1. Load clean speech (bundled librosa example, resampled to 16 kHz).
2. Load real recorded wind noise (mobile phone recordings, strong wind) from
   the Zenodo wind noise dataset (record 6687982).
3. Mix speech + wind at a target SNR.
4. Suppress with classical DSP and neural methods:
   - High-pass filter (wind energy is concentrated below ~300 Hz)
   - Spectral gating (noisereduce, non-stationary mode)
   - Meta Denoiser (DNS64, facebookresearch/denoiser)
   - DeepFilterNet (via .venv-dfn subprocess, see dfn_enhance.py)
5. Report SI-SDR / PESQ-WB / STOI and save wavs + spectrogram plots.
"""

import subprocess
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import noisereduce as nr
import soundfile as sf
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn
from scipy.signal import butter, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FS = 16000
SNR_DB = 0
WIND_DIR = ROOT / "data" / "mobilephone_wind_noise" / "strong_wind"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
DFN_PYTHON = ROOT / ".venv-dfn" / "bin" / "python"


def load_real_wind(n_samples, session="009"):
    """Concatenate consecutive 3-second chunks of one phone recording session."""
    files = sorted(WIND_DIR.glob(f"{session}_*.wav"))
    if not files:
        raise FileNotFoundError(f"no wind files for session {session} in {WIND_DIR}")
    chunks = []
    total = 0
    for f in files:
        x, fs = sf.read(f)
        assert fs == FS, f"expected {FS} Hz, got {fs}"
        chunks.append(x)
        total += len(x)
        if total >= n_samples:
            break
    if total < n_samples:
        raise ValueError(f"session {session} too short: {total} < {n_samples}")
    return np.concatenate(chunks)[:n_samples]


def si_sdr(reference, estimate):
    """Scale-invariant signal-to-distortion ratio in dB."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / np.dot(reference, reference)
    target = alpha * reference
    noise = estimate - target
    return 10 * np.log10(np.sum(target**2) / np.sum(noise**2))


def evaluate(reference, estimate):
    """Return dict of SI-SDR (dB), PESQ-WB, and STOI."""
    return {
        "si_sdr": si_sdr(reference, estimate),
        "pesq": pesq_fn(FS, reference, estimate, "wb"),
        "stoi": stoi_fn(reference, estimate, FS, extended=False),
    }


def mix_at_snr(speech, noise, snr_db):
    """Scale noise so that speech/noise power ratio equals snr_db."""
    p_speech = np.mean(speech**2)
    p_noise = np.mean(noise**2)
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return speech + gain * noise, gain * noise


def highpass(x, fs, cutoff=150, order=6):
    """Zero-phase Butterworth high-pass filter."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def spectral_gate(x, fs):
    """Non-stationary spectral gating via noisereduce."""
    return nr.reduce_noise(y=x, sr=fs, stationary=False, prop_decrease=0.95)


def meta_denoiser(x, fs):
    """Meta Denoiser DNS64 (facebookresearch/denoiser), causal, 16 kHz."""
    model = pretrained.dns64()
    wav = torch.from_numpy(x).float().unsqueeze(0)
    wav = convert_audio(wav, fs, model.sample_rate, model.chin)
    with torch.no_grad():
        out = model(wav.unsqueeze(0))[0]
    return out.squeeze(0).numpy()[: len(x)]


def deepfilternet(x, fs):
    """DeepFilterNet3 via subprocess in the .venv-dfn (py3.11) environment."""
    tmp_in = RESULTS / "_dfn_in.wav"
    tmp_out = RESULTS / "_dfn_out.wav"
    sf.write(tmp_in, x, fs)
    subprocess.run([str(DFN_PYTHON), str(ROOT / "src" / "dfn_enhance.py"),
                    str(tmp_in), str(tmp_out)], check=True)
    out, out_fs = sf.read(tmp_out)
    assert out_fs == fs
    tmp_in.unlink()
    tmp_out.unlink()
    return out[: len(x)]


def plot_spectrograms(signals, fs, out_path):
    """Save a grid of spectrograms, one per (title, signal) pair."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 3 * len(signals)),
                             sharex=True)
    n_fft, hop = 512, 128
    # common dB reference so panels are directly comparable
    ref = max(np.max(np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)))
              for _, x in signals)
    for ax, (title, x) in zip(axes, signals):
        stft = librosa.stft(x, n_fft=n_fft, hop_length=hop)
        db = librosa.amplitude_to_db(np.abs(stft), ref=ref)
        img = librosa.display.specshow(db, sr=fs, hop_length=hop,
                                       x_axis="time", y_axis="hz",
                                       ax=ax, cmap="magma", vmin=-80, vmax=0)
        ax.set_title(title)
        ax.set_ylim(0, 8000)
    fig.colorbar(img, ax=axes, format="%+2.0f dB")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    # 1. Clean speech
    speech, _ = librosa.load(librosa.example("libri1"), sr=FS, duration=8)
    speech = speech / np.max(np.abs(speech))

    # 2. Real recorded wind noise
    wind = load_real_wind(len(speech))

    # 3. Mix
    noisy, wind_scaled = mix_at_snr(speech, wind, SNR_DB)

    # 4. Suppress
    methods = {
        "highpass": lambda: highpass(noisy, FS),
        "spectral_gate": lambda: spectral_gate(noisy, FS),
        "meta_denoiser": lambda: meta_denoiser(noisy, FS),
        "deepfilternet": lambda: deepfilternet(noisy, FS),
    }
    enhanced = {}
    for name, fn in methods.items():
        print(f"running {name} ...")
        enhanced[name] = fn()

    # 5. Metrics + outputs
    peak = max(np.max(np.abs(noisy)),
               *(np.max(np.abs(x)) for x in enhanced.values()))
    sf.write(RESULTS / "clean.wav", speech / peak, FS)
    sf.write(RESULTS / "wind_only.wav", wind_scaled / peak, FS)
    sf.write(RESULTS / "noisy.wav", noisy / peak, FS)

    rows = [("noisy (no processing)", evaluate(speech, noisy))]
    for name, x in enhanced.items():
        sf.write(RESULTS / f"enhanced_{name}.wav", x / peak, FS)
        rows.append((name, evaluate(speech, x)))

    print(f"\nInput SNR: {SNR_DB} dB (real strong wind, phone recording)")
    print(f"{'method':24s} {'SI-SDR':>8s} {'PESQ':>6s} {'STOI':>6s}")
    with open(RESULTS / "metrics.csv", "w") as f:
        f.write("method,si_sdr_db,pesq_wb,stoi\n")
        for name, m in rows:
            print(f"{name:24s} {m['si_sdr']:8.2f} {m['pesq']:6.2f} {m['stoi']:6.3f}")
            f.write(f"{name},{m['si_sdr']:.2f},{m['pesq']:.2f},{m['stoi']:.3f}\n")

    plot_spectrograms(
        [("Clean speech", speech),
         ("Noisy (speech + real wind, 0 dB SNR)", noisy),
         ("High-pass 150 Hz", enhanced["highpass"]),
         ("Spectral gating (noisereduce)", enhanced["spectral_gate"]),
         ("Meta Denoiser (DNS64)", enhanced["meta_denoiser"]),
         ("DeepFilterNet", enhanced["deepfilternet"])],
        FS, PLOTS / "spectrograms.png")
    print(f"\nWavs -> {RESULTS}/  |  Plot -> {PLOTS}/spectrograms.png")


if __name__ == "__main__":
    main()
