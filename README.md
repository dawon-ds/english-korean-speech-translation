# English–Korean Speech Translation Pipeline

An end-to-end pipeline that converts **English speech into Korean speech** through speech recognition, machine translation, and neural text-to-speech.

[Portfolio](https://app.notion.com/p/3e968564df5a81f0b538ed4ad190e11f)

## Project Overview

A personal project connecting heterogeneous speech and language models into one inference workflow. The goal is to translate English audio into Korean audio while handling the interfaces between speech recognition, translation, Korean text preprocessing, and synthesis.

**Technologies:** Python, PyTorch, Hugging Face Transformers, Whisper, g2pK, Jamo, FastSpeech2, VocGAN.

## Pipeline

```text
English Speech → Whisper → English Text → mBART-50 → Korean Text
               → g2pK/Jamo → FastSpeech2 → Mel-Spectrogram → VocGAN → Korean Speech
```

## Implementation

- Integrated Whisper `base.en` for English speech recognition.
- Added English-to-Korean translation with mBART-50.
- Connected Korean G2P/Jamo preprocessing to FastSpeech2.
- Integrated FastSpeech2 and pretrained VocGAN into one inference flow.
- Added CPU/GPU device selection.
- Replaced the original machine-specific KSS path with `KSS_DATA_PATH`.

## Project Structure

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

## Main Components

| Stage | Component |
| --- | --- |
| Speech-to-Text | Whisper `base.en` |
| Translation | mBART-50 |
| Korean preprocessing | g2pK + Jamo |
| Text-to-Speech | FastSpeech2 |
| Vocoder | VocGAN |
| Framework | PyTorch |

## Model Configuration

| FastSpeech2 setting | Value |
| --- | ---: |
| Sampling rate | 22,050 Hz |
| Mel channels | 80 |
| Encoder / decoder layers | 4 / 4 |
| Hidden size | 256 |
| Inference checkpoint step | 350000 |

FastSpeech2 training was monitored through F0, duration, energy, mel, mel-postnet, and total loss in the project materials.

## Inference Example

| Stage | Example |
| --- | --- |
| English input | “She looked into the mirror and saw another person.” |
| Korean output text | “그녀는 거울을 들여다보고 다른 사람을 봤습니다.” |

The portfolio includes input/output audio and the synthesized spectrogram for this example. It demonstrates the connected inference workflow; it is not a quantitative benchmark.

## Running

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

The supplied pipeline uses FastSpeech2 checkpoint step `350000`.

## Results & Limitations

The project materials demonstrate Korean speech generation from English input using the connected pipeline. They do not provide quantitative STT accuracy, translation BLEU, TTS quality scores, or latency benchmarks.

Errors can propagate from speech recognition through translation to synthesis. A fresh checkout also requires external checkpoints and preprocessing artifacts before inference can run. The documented example has not been rerun as part of this documentation update.

## Review

The main contribution is the integration of speech recognition, translation, Korean pronunciation preprocessing, and speech synthesis into a modular workflow. Separating these stages makes each model interface easier to inspect and reuse, while highlighting the importance of compatible text representations and audio settings.

## Notes

This repository is a cleaned portfolio version of the original project. Generated audio, datasets, checkpoints, and pretrained vocoder weights are excluded for portability. The supplied project files did not contain quantitative STT, translation, or TTS benchmark metrics, so no unsupported metrics are claimed here.
