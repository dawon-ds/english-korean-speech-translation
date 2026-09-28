import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hparams as hp
import utils
from dataset import Dataset
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(step):
    """Load a FastSpeech2 checkpoint for evaluation."""
    model = nn.DataParallel(FastSpeech2()).to(DEVICE)
    checkpoint_path = os.path.join(
        hp.checkpoint_path,
        f"checkpoint_{step}.pth.tar",
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _to_tensor(batch, key, dtype):
    return torch.from_numpy(batch[key]).to(dtype=dtype, device=DEVICE)


def evaluate(model, step, vocoder=None):
    """Evaluate FastSpeech2 on the validation split."""
    dataset = Dataset("val.txt", sort=False)
    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size**2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    criterion = FastSpeech2Loss().to(DEVICE)
    loss_values = [[] for _ in range(5)]

    for batches in loader:
        for batch in batches:
            text = _to_tensor(batch, "text", torch.long)
            mel = _to_tensor(batch, "mel_target", torch.float)
            duration = _to_tensor(batch, "D", torch.long)
            log_duration = _to_tensor(batch, "log_D", torch.float)
            f0 = _to_tensor(batch, "f0", torch.float)
            energy = _to_tensor(batch, "energy", torch.float)
            src_len = _to_tensor(batch, "src_len", torch.long)
            mel_len = _to_tensor(batch, "mel_len", torch.long)

            with torch.no_grad():
                output = model(
                    text,
                    src_len,
                    mel_len,
                    duration,
                    f0,
                    energy,
                    int(np.max(batch["src_len"])),
                    int(np.max(batch["mel_len"])),
                )
                losses = criterion(
                    output[2],
                    log_duration,
                    output[3],
                    f0,
                    output[4],
                    energy,
                    output[0],
                    output[1],
                    mel,
                    ~output[5],
                    ~output[6],
                )

            for values, loss in zip(loss_values, losses):
                values.append(loss.item())

    means = [sum(values) / len(values) for values in loss_values]
    print(
        f"FastSpeech2 Step {step}: "
        f"mel={means[0]:.4f}, postnet={means[1]:.4f}, "
        f"duration={means[2]:.4f}, f0={means[3]:.4f}, "
        f"energy={means[4]:.4f}"
    )

    model.train()
    return means[2], means[3], means[4], means[0], means[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=30000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fastspeech2 = load_model(args.step)
    vocoder = (
        utils.get_vocgan(hp.vocoder_pretrained_model_path)
        if hp.vocoder == "vocgan"
        else None
    )
    evaluate(fastspeech2, args.step, vocoder)
