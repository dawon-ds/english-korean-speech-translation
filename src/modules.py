from collections import OrderedDict

import torch
import torch.nn as nn

import hparams as hp
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Predict and apply duration, pitch, and energy information."""

    def __init__(self):
        super().__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.pitch_embedding_producer = Conv(
            1, hp.encoder_hidden, kernel_size=9, padding=4, bias=False
        )
        self.energy_embedding_producer = Conv(
            1, hp.encoder_hidden, kernel_size=9, padding=4, bias=False
        )

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        max_len=None,
        dur_pitch_energy_aug=None,
        f0_stat=None,
        energy_stat=None,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction = self.pitch_predictor(x, src_mask)

        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding_producer(
                pitch_target.unsqueeze(2)
            )
        else:
            pitch_prediction = utils.de_norm(
                pitch_prediction, f0_stat[0], f0_stat[1]
            )
            pitch_prediction *= dur_pitch_energy_aug[1]
            pitch_prediction = utils.standard_norm_torch(
                pitch_prediction, f0_stat[0], f0_stat[1]
            )
            pitch_embedding = self.pitch_embedding_producer(
                pitch_prediction.unsqueeze(2)
            )

        energy_prediction = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding_producer(
                energy_target.unsqueeze(2)
            )
        else:
            energy_prediction = utils.de_norm(
                energy_prediction, energy_stat[0], energy_stat[1]
            )
            energy_prediction *= dur_pitch_energy_aug[2]
            energy_prediction = utils.standard_norm_torch(
                energy_prediction, energy_stat[0], energy_stat[1]
            )
            energy_embedding = self.energy_embedding_producer(
                energy_prediction.unsqueeze(2)
            )

        x = x + pitch_embedding + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration = torch.clamp(
                torch.round(
                    torch.exp(log_duration_prediction) - hp.log_offset
                )
                * dur_pitch_energy_aug[0],
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)

        return (
            x,
            log_duration_prediction,
            pitch_prediction,
            energy_prediction,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Expand encoder states according to predicted phoneme durations."""

    @staticmethod
    def expand(sequence, durations):
        expanded = [
            vector.expand(int(durations[index].item()), -1)
            for index, vector in enumerate(sequence)
        ]
        return torch.cat(expanded, dim=0)

    def forward(self, x, duration, max_len=None):
        outputs = []
        mel_lengths = []

        for sequence, target_duration in zip(x, duration):
            expanded = self.expand(sequence, target_duration)
            outputs.append(expanded)
            mel_lengths.append(expanded.shape[0])

        output = utils.pad(outputs, max_len)
        return output, torch.LongTensor(mel_lengths).to(DEVICE)


class VariancePredictor(nn.Module):
    """Convolutional predictor used for duration, pitch, and energy."""

    def __init__(self):
        super().__init__()
        filter_size = hp.variance_predictor_filter_size
        kernel_size = hp.variance_predictor_kernel_size
        dropout = hp.variance_predictor_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            hp.encoder_hidden,
                            filter_size,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(filter_size)),
                    ("dropout_1", nn.Dropout(dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            filter_size,
                            filter_size,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(filter_size)),
                    ("dropout_2", nn.Dropout(dropout)),
                ]
            )
        )
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, x, mask):
        output = self.linear_layer(self.conv_layer(x)).squeeze(-1)
        return output.masked_fill(mask, 0.0) if mask is not None else output


class Conv(nn.Module):
    """Conv1d wrapper that accepts tensors shaped as [batch, time, channel]."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        return self.conv(x).contiguous().transpose(1, 2)
