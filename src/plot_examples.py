"""Plot input/output waveforms + spectrograms for all algorithms on 2 examples.

Example 1: libri1 speech + strong wind (session 009)
Example 2: libri2 speech + normal wind (session 000)

For each example, saves a grid figure (rows = clean/noisy/each algorithm,
columns = waveform, spectrogram) to plots/, with metrics in the row titles.
"""

from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wind_noise_demo import (FS, SNR_DB, ROOT, PLOTS, evaluate, mix_at_snr,
                             highpass, spectral_gate, meta_denoiser,
                             deepfilternet)

EXAMPLES = [
    {"name": "example1_strong_wind", "speech": "libri1",
     "wind_dir": "strong_wind", "session": "009",
     "title": "libri1 + strong wind (session 009)"},
    {"name": "example2_normal_wind", "speech": "libri2",
     "wind_dir": "normal_wind", "session": "000",
     "title": "libri2 + normal wind (session 000)"},
]


def load_wind(wind_dir, session, n_samples):
    """Concatenate consecutive 3 s chunks of one phone recording session."""
    files = sorted((ROOT / "data" / "mobilephone_wind_noise" / wind_dir)
                   .glob(f"{session}_*.wav"))
    chunks = []
    total = 0
    for f in files:
        x, fs = sf.read(f)
        assert fs == FS
        chunks.append(x)
        total += len(x)
        if total >= n_samples:
            break
    if total < n_samples:
        raise ValueError(f"{wind_dir}/{session}: only {total} samples")
    return np.concatenate(chunks)[:n_samples]


def plot_grid(signals, fs, suptitle, out_path):
    """Rows = (title, signal); columns = waveform, spectrogram."""
    n = len(signals)
    fig, axes = plt.subplots(n, 2, figsize=(16, 2.4 * n), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})
    t = np.arange(len(signals[0][1])) / fs
    ymax = max(np.max(np.abs(x)) for _, x in signals) * 1.05
    n_fft, hop = 512, 128
    ref = max(np.max(np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)))
              for _, x in signals)
    for row, (title, x) in enumerate(signals):
        ax_w, ax_s = axes[row]
        ax_w.plot(t, x, linewidth=0.4, color="#1f77b4")
        ax_w.set_ylim(-ymax, ymax)
        ax_w.set_ylabel("amplitude")
        ax_w.set_title(f"{title} — waveform", fontsize=10, loc="left")
        db = librosa.amplitude_to_db(
            np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)), ref=ref)
        img = librosa.display.specshow(db, sr=fs, hop_length=hop,
                                       x_axis="time", y_axis="hz",
                                       ax=ax_s, cmap="magma", vmin=-80, vmax=0)
        ax_s.set_ylim(0, 8000)
        ax_s.set_title(f"{title} — spectrogram", fontsize=10, loc="left")
    axes[-1][0].set_xlabel("Time (s)")
    fig.colorbar(img, ax=axes[:, 1], format="%+2.0f dB", pad=0.01)
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def label(name, m):
    return (f"{name} (SI-SDR {m['si_sdr']:.1f} dB, "
            f"PESQ {m['pesq']:.2f}, STOI {m['stoi']:.2f})")


def main():
    PLOTS.mkdir(exist_ok=True)
    for ex in EXAMPLES:
        print(f"=== {ex['title']} ===")
        speech, _ = librosa.load(librosa.example(ex["speech"]), sr=FS,
                                 duration=8)
        speech = speech / np.max(np.abs(speech))
        wind = load_wind(ex["wind_dir"], ex["session"], len(speech))
        noisy, _ = mix_at_snr(speech, wind, SNR_DB)

        methods = {
            "High-pass 150 Hz": highpass(noisy, FS),
            "Spectral gating": spectral_gate(noisy, FS),
            "Meta Denoiser (DNS64)": meta_denoiser(noisy, FS),
            "DeepFilterNet": deepfilternet(noisy, FS),
        }

        signals = [("Clean speech", speech),
                   (label("Noisy input", evaluate(speech, noisy)), noisy)]
        for name, x in methods.items():
            m = evaluate(speech, x)
            print(f"  {name:24s} SI-SDR {m['si_sdr']:6.2f}  "
                  f"PESQ {m['pesq']:.2f}  STOI {m['stoi']:.3f}")
            signals.append((label(name, m), x))

        out = PLOTS / f"{ex['name']}.png"
        plot_grid(signals, FS, f"Wind noise suppression — {ex['title']}, "
                  f"{SNR_DB} dB SNR", out)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
