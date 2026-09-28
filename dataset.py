import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import hparams
from utils import pad_1D, pad_2D, process_meta, standard_norm
from text import text_to_sequence

class Dataset(TorchDataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.text = process_meta(os.path.join(hparams.preprocessed_path, filename))
        self.mean_mel, self.std_mel = np.load(os.path.join(hparams.preprocessed_path, "mel_stat.npy"))
        self.mean_f0, self.std_f0 = np.load(os.path.join(hparams.preprocessed_path, "f0_stat.npy"))
        self.mean_energy, self.std_energy = np.load(os.path.join(hparams.preprocessed_path, "energy_stat.npy"))
        self.sort = sort

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename, text = self.basename[idx], self.text[idx]
        def load(kind, tag):
            return np.load(os.path.join(hparams.preprocessed_path, kind, f"{hparams.dataset}-{tag}-{basename}.npy"))
        return {"id": basename, "text": np.array(text_to_sequence(text, [])),
                "mel_target": load("mel", "mel"), "D": load("alignment", "ali"),
                "f0": load("f0", "f0"), "energy": load("energy", "energy")}

    def reprocess(self, batch, cut_list):
        ids = [batch[i]["id"] for i in cut_list]
        texts = [batch[i]["text"] for i in cut_list]
        mels = [standard_norm(batch[i]["mel_target"], self.mean_mel, self.std_mel, is_mel=True) for i in cut_list]
        durations = [batch[i]["D"] for i in cut_list]
        f0s = [standard_norm(batch[i]["f0"], self.mean_f0, self.std_f0) for i in cut_list]
        energies = [standard_norm(batch[i]["energy"], self.mean_energy, self.std_energy) for i in cut_list]
        src_len = np.array([x.shape[0] for x in texts])
        mel_len = np.array([x.shape[0] for x in mels])
        durations = pad_1D(durations)
        return {"id": ids, "text": pad_1D(texts), "mel_target": pad_2D(mels),
                "D": durations, "log_D": np.log(durations + hparams.log_offset),
                "f0": pad_1D(f0s), "energy": pad_1D(energies),
                "src_len": src_len, "mel_len": mel_len}

    def collate_fn(self, batch):
        order = np.argsort(-np.array([d["text"].shape[0] for d in batch]))
        n = int(math.sqrt(len(batch)))
        return [self.reprocess(batch, order[i*n:(i+1)*n]) for i in range(n)]
