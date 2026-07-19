"""DeepFilterNet enhancement runner (executed with .venv-dfn python 3.11).

Usage: python dfn_enhance.py <in.wav> <out.wav>
Loads input, enhances at DeepFilterNet's 48 kHz, resamples back to input rate.
"""

import sys

import soundfile as sf
import torch
import torchaudio
from df.enhance import enhance, init_df


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    x, in_sr = sf.read(in_path, dtype="float32")

    model, df_state, _ = init_df(log_level="ERROR")
    audio = torch.from_numpy(x).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, in_sr, df_state.sr())
    with torch.no_grad():
        out = enhance(model, df_state, audio)
    out = torchaudio.functional.resample(out, df_state.sr(), in_sr)
    sf.write(out_path, out.squeeze(0).numpy(), in_sr)


if __name__ == "__main__":
    main()
