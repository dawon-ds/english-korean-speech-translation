import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import hparams as hp
from vocoder.vocgan_generator import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SILENCE_PHONES = {"sil", "sp", "spn"}


def get_alignment(tier):
    """Extract phones, frame durations, and voiced time boundaries."""
    phones = []
    durations = []
    start = 0
    end = 0
    end_index = 0

    for interval in tier._objects:
        start_time = interval.start_time
        end_time = interval.end_time
        phone = interval.text

        if not phones and phone in SILENCE_PHONES:
            continue
        if not phones:
            start = start_time

        phones.append(phone)
        if phone not in SILENCE_PHONES:
            end = end_time
            end_index = len(phones)

        start_frame = int(
            start_time * hp.sampling_rate / hp.hop_length
        )
        end_frame = int(end_time * hp.sampling_rate / hp.hop_length)
        durations.append(end_frame - start_frame)

    return (
        phones[:end_index],
        np.array(durations[:end_index]),
        start,
        end,
    )


def process_meta(path):
    """Read preprocessed metadata in basename|text format."""
    with open(path, "r", encoding="utf-8") as file:
        rows = [
            line.strip().split("|", 1)
            for line in file
            if line.strip()
        ]
    return [row[0] for row in rows], [row[1] for row in rows]


def get_param_num(model):
    return sum(parameter.numel() for parameter in model.parameters())


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(max_len, device=lengths.device)
    ids = ids.unsqueeze(0).expand(lengths.shape[0], -1)
    return ids >= lengths.unsqueeze(1)


def get_vocgan(
    ckpt_path,
    n_mel_channels=hp.n_mel_channels,
    generator_ratio=None,
    n_residual_layers=4,
    mult=256,
    out_channels=1,
):
    """Load a pretrained VocGAN generator."""
    if generator_ratio is None:
        generator_ratio = [4, 4, 2, 2, 2, 2]

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = Generator(
        n_mel_channels,
        n_residual_layers,
        ratios=generator_ratio,
        mult=mult,
        out_band=out_channels,
    )
    model.load_state_dict(checkpoint["model_g"])
    return model.to(DEVICE).eval()


def vocgan_infer(mel, vocoder, path):
    """Generate a waveform from a mel-spectrogram with VocGAN."""
    with torch.no_grad():
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        audio = vocoder.infer(mel).squeeze()
        audio = hp.max_wav_value * audio[: -(hp.hop_length * 10)]
        audio = audio.clamp(
            -hp.max_wav_value,
            hp.max_wav_value - 1,
        )
        wavfile.write(
            path,
            hp.sampling_rate,
            audio.short().cpu().numpy(),
        )


def pad_1D(inputs, pad_value=0):
    max_len = max(len(item) for item in inputs)
    return np.stack(
        [
            np.pad(
                item,
                (0, max_len - item.shape[0]),
                constant_values=pad_value,
            )
            for item in inputs
        ]
    )


def pad_2D(inputs, maxlen=None):
    max_len = maxlen or max(item.shape[0] for item in inputs)
    return np.stack(
        [
            np.pad(
                item,
                ((0, max_len - item.shape[0]), (0, 0)),
                constant_values=0,
            )
            for item in inputs
        ]
    )


def pad(inputs, mel_max_length=None):
    max_len = mel_max_length or max(item.size(0) for item in inputs)
    outputs = []

    for item in inputs:
        if item.dim() == 1:
            outputs.append(F.pad(item, (0, max_len - item.size(0))))
        else:
            outputs.append(
                F.pad(item, (0, 0, 0, max_len - item.size(0)))
            )
    return torch.stack(outputs)


def standard_norm(x, mean, std, is_mel=False):
    if not is_mel:
        x = remove_outlier(x)
    zero_indices = np.where(x == 0.0)
    x = (x - mean) / std
    x[zero_indices] = 0.0
    return x


def standard_norm_torch(x, mean, std):
    zero_indices = torch.where(x == 0.0)
    x = (x - mean) / std
    x[zero_indices] = 0.0
    return x


def de_norm(x, mean, std):
    zero_indices = torch.where(x == 0.0)
    x = mean + std * x
    x[zero_indices] = 0.0
    return x


def remove_outlier(x):
    lower_quartile, upper_quartile = np.percentile(x, [25, 75])
    iqr = upper_quartile - lower_quartile
    lower_bound = lower_quartile - 1.5 * iqr
    upper_bound = upper_quartile + 1.5 * iqr
    outliers = np.logical_or(x <= lower_bound, x >= upper_bound)
    x[outliers] = 0.0
    return x


def average_by_duration(values, durations):
    boundaries = np.cumsum(np.pad(durations, (1, 0)))
    output = np.zeros(len(durations), dtype=np.float32)

    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        segment = values[start:end]
        segment = segment[segment != 0]
        output[index] = np.mean(segment) if len(segment) else 0.0
    return output


def plot_data(data, titles=None, filename=None):
    """Save spectrogram visualizations used during synthesis."""
    _, axes = plt.subplots(len(data), 1, squeeze=False)
    titles = titles or [None] * len(data)

    for index, (spectrogram, _pitch, _energy) in enumerate(data):
        axes[index][0].imshow(
            spectrogram,
            origin="lower",
            aspect="auto",
        )
        axes[index][0].set_title(titles[index] or "")

    plt.savefig(filename, dpi=200)
    plt.close()
