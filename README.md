# English–Korean Speech Translation Pipeline

**English speech → Korean speech**, integrating speech recognition, translation, and neural speech synthesis in a single inference workflow.

## Project Overview

A personal project focused on connecting pretrained models through compatible text, pronunciation, and audio representations.

## Approach

### Pipeline

```text
English Speech → Whisper → English Text → mBART-50 → Korean Text
               → g2pK/Jamo → FastSpeech2 → Mel-Spectrogram → VocGAN → Korean Speech
```

- **STT:** Whisper `base.en`
- **Translation:** `facebook/mbart-large-50-many-to-many-mmt`
- **Korean pronunciation processing:** g2pK and Jamo decomposition
- **Speech synthesis:** FastSpeech2 and VocGAN

The modules are connected by `src/pipeline.py`; CUDA availability determines CPU/GPU selection.

### FastSpeech2 Configuration

| FastSpeech2 setting | Value |
| --- | ---: |
| Sampling rate | 22,050 Hz |
| Mel channels | 80 |
| Encoder / decoder layers | 4 / 4 |
| Hidden size | 256 |
| Inference checkpoint step | 350000 |

Training records include F0, duration, energy, mel, mel-postnet, and total loss monitoring.

## Results

A documented inference example produced Korean speech from English audio; input/output WAV samples and the synthesized spectrogram are available in the [portfolio](https://app.notion.com/p/3e968564df5a81f0b538ed4ad190e11f). Quantitative STT, translation, speech-quality, and latency scores are not available.

## Repository Structure

```text
.
├── src/
│   ├── pipeline.py          # End-to-end inference
│   ├── stt.py               # Whisper STT
│   ├── translate.py         # mBART EN → KO translation
│   ├── synthesize.py        # Speech synthesis entry point
│   ├── fastspeech2.py       # FastSpeech2 model
│   ├── modules.py           # Variance adaptor and predictors
│   ├── transformer/         # Encoder/decoder and FFT blocks
│   ├── vocoder/             # VocGAN generator
│   ├── text/                # Korean text preprocessing
│   ├── audio/               # STFT and mel utilities
│   ├── data/                # KSS preprocessing
│   ├── dataset.py           # Training dataset loader
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation pipeline
│   ├── loss.py
│   ├── optimizer.py
│   └── hparams.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage

### Inference Example

| Stage | Example |
| --- | --- |
| English input | “She looked into the mirror and saw another person.” |
| Korean output text | “그녀는 거울을 들여다보고 다른 사람을 봤습니다.” |

The pipeline requires preprocessed KSS artifacts, a FastSpeech2 checkpoint, and a pretrained VocGAN checkpoint. Large datasets and model weights are intentionally excluded.

From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

Place the required artifacts at the paths configured in `src/hparams.py`:

| Artifact | Default path |
| --- | --- |
| Input English WAV | `input_voice.wav` |
| FastSpeech2 checkpoint | `ckpt/kss/checkpoint_350000.pth.tar` |
| Preprocessed KSS statistics | `preprocessed/kss/mel_stat.npy`, `f0_stat.npy`, `energy_stat.npy` |
| VocGAN weights | `vocoder/pretrained_models/vocgan_kss_pretrained_model_epoch_4500.pt` |

For PowerShell:

```powershell
$env:KSS_DATA_PATH="C:/path/to/kss"
python src/pipeline.py
```

For Bash:

```bash
export KSS_DATA_PATH="/path/to/kss"
python src/pipeline.py
```

`KSS_DATA_PATH` controls the source KSS dataset location; it does not change the checkpoint or preprocessed-artifact paths. Edit `src/hparams.py` if those files are stored elsewhere. The entry point uses `input_voice.wav`; change the argument in `src/pipeline.py` to use another input.

Synthesis writes these outputs under `results/`:

- `_vocgan_output_voi.wav` — VocGAN speech output
- `_griffin_lim_output_voi.wav` — Griffin–Lim reconstruction
- `_output_voi.png` — synthesized spectrogram, F0, and energy

The current synthesis function truncates the output label to ten characters, which produces the `output_voi` filenames.

### Data & Artifacts

The full KSS data, generated audio, checkpoints, and vocoder weights are not included in this repository.

## Limitations

Errors can propagate from speech recognition through translation to synthesis. A fresh checkout also requires external checkpoints and preprocessing artifacts before inference can run.

## Data and Third-party Resources

- **Third-party models:** Whisper, mBART-50, FastSpeech2 and VocGAN. Check the license and permitted uses of each particular codebase, checkpoint and dataset; trained weights and KSS audio are not redistributed.
Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
