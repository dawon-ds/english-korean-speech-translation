import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hparams as hp
import utils
from dataset import Dataset
from evaluate import evaluate
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from optimizer import ScheduledOptim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(batch, key, dtype):
    """Move a NumPy batch field to the active PyTorch device."""
    return torch.from_numpy(batch[key]).to(dtype=dtype, device=DEVICE)


def load_checkpoint(model, optimizer, restore_step):
    """Restore model and optimizer state when a checkpoint is available."""
    if not restore_step:
        return

    checkpoint_path = os.path.join(
        hp.checkpoint_path,
        f"checkpoint_{restore_step}.pth.tar",
    )
    if not os.path.exists(checkpoint_path):
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train_batch(model, criterion, batch):
    """Run a forward pass and return the FastSpeech2 loss terms."""
    text = _to_tensor(batch, "text", torch.long)
    mel = _to_tensor(batch, "mel_target", torch.float)
    duration = _to_tensor(batch, "D", torch.long)
    log_duration = _to_tensor(batch, "log_D", torch.float)
    f0 = _to_tensor(batch, "f0", torch.float)
    energy = _to_tensor(batch, "energy", torch.float)
    src_len = _to_tensor(batch, "src_len", torch.long)
    mel_len = _to_tensor(batch, "mel_len", torch.long)

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

    return criterion(
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


def main(args):
    torch.manual_seed(0)

    dataset = Dataset("train.txt")
    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size**2,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    model = nn.DataParallel(FastSpeech2()).to(DEVICE)
    print(
        "Number of FastSpeech2 Parameters:",
        utils.get_param_num(model),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=hp.betas,
        eps=hp.eps,
        weight_decay=hp.weight_decay,
    )
    scheduler = ScheduledOptim(
        optimizer,
        hp.decoder_hidden,
        hp.n_warm_up_step,
        args.restore_step,
    )
    criterion = FastSpeech2Loss().to(DEVICE)

    os.makedirs(hp.checkpoint_path, exist_ok=True)
    os.makedirs(hp.log_path, exist_ok=True)
    load_checkpoint(model, optimizer, args.restore_step)

    vocoder = (
        utils.get_vocgan(hp.vocoder_pretrained_model_path)
        if hp.vocoder == "vocgan"
        else None
    )
    logger = SummaryWriter(os.path.join(hp.log_path, "train"))
    model.train()

    for epoch in range(hp.epochs):
        for loader_index, batches in enumerate(loader):
            for batch_index, batch in enumerate(batches):
                step = (
                    loader_index * hp.batch_size
                    + batch_index
                    + args.restore_step
                    + epoch * len(loader) * hp.batch_size
                    + 1
                )

                losses = train_batch(model, criterion, batch)
                total_loss = sum(losses) / hp.acc_steps
                total_loss.backward()

                if step % hp.acc_steps:
                    continue

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    hp.grad_clip_thresh,
                )
                scheduler.step_and_update_lr()
                scheduler.zero_grad()

                logger.add_scalar(
                    "Loss/total_loss",
                    float(total_loss.item() * hp.acc_steps),
                    step,
                )

                if step % hp.save_step == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    checkpoint_path = os.path.join(
                        hp.checkpoint_path,
                        f"checkpoint_{step}.pth.tar",
                    )
                    torch.save(checkpoint, checkpoint_path)

                if step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        evaluate(model, step, vocoder)
                    model.train()

    logger.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
