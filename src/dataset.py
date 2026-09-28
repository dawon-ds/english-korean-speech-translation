import math
import os

import numpy as np
from torch.utils.data import Dataset as TorchDataset

import hparams
from text import text_to_sequence
from utils import pad_1D, pad_2D, process_meta, standard_norm


class Dataset(TorchDataset):
    """Load preprocessed KSS features for FastSpeech2 training."""

    def __init__(self, filename="train.txt", sort=True):
        metadata_path = os.path.join(hparams.preprocessed_path, filename)
        self.basename, self.text = process_meta(metadata_path)

        self.mean_mel, self.std_mel = self._load_stats("mel_stat.npy")
        self.mean_f0, self.std_f0 = self._load_stats("f0_stat.npy")
        self.mean_energy, self.std_energy = self._load_stats(
            "energy_stat.npy"
        )
        self.sort = sort

    @staticmethod
    def _load_stats(filename):
        return np.load(os.path.join(hparams.preprocessed_path, filename))

    @staticmethod
    def _feature_path(kind, tag, basename):
        filename = f"{hparams.dataset}-{tag}-{basename}.npy"
        return os.path.join(hparams.preprocessed_path, kind, filename)

    def _load_feature(self, kind, tag, basename):
        return np.load(self._feature_path(kind, tag, basename))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        basename = self.basename[index]
        text = self.text[index]

        return {
            "id": basename,
            "text": np.array(text_to_sequence(text, [])),
            "mel_target": self._load_feature("mel", "mel", basename),
            "D": self._load_feature("alignment", "ali", basename),
            "f0": self._load_feature("f0", "f0", basename),
            "energy": self._load_feature("energy", "energy", basename),
        }

    def reprocess(self, batch, indices):
        ids = [batch[i]["id"] for i in indices]
        texts = [batch[i]["text"] for i in indices]
        durations = [batch[i]["D"] for i in indices]

        mels = [
            standard_norm(
                batch[i]["mel_target"],
                self.mean_mel,
                self.std_mel,
                is_mel=True,
            )
            for i in indices
        ]
        f0s = [
            standard_norm(batch[i]["f0"], self.mean_f0, self.std_f0)
            for i in indices
        ]
        energies = [
            standard_norm(
                batch[i]["energy"],
                self.mean_energy,
                self.std_energy,
            )
            for i in indices
        ]

        src_len = np.array([text.shape[0] for text in texts])
        mel_len = np.array([mel.shape[0] for mel in mels])
        durations = pad_1D(durations)

        return {
            "id": ids,
            "text": pad_1D(texts),
            "mel_target": pad_2D(mels),
            "D": durations,
            "log_D": np.log(durations + hparams.log_offset),
            "f0": pad_1D(f0s),
            "energy": pad_1D(energies),
            "src_len": src_len,
            "mel_len": mel_len,
        }

    def collate_fn(self, batch):
        """Sort samples by text length and build mini-batches."""
        lengths = np.array([sample["text"].shape[0] for sample in batch])
        order = np.argsort(-lengths) if self.sort else np.arange(len(batch))

        group_size = int(math.sqrt(len(batch)))
        return [
            self.reprocess(
                batch,
                order[i * group_size : (i + 1) * group_size],
            )
            for i in range(group_size)
        ]
