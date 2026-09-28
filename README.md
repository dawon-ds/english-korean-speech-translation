# English-to-Korean Speech Translation Pipeline

An end-to-end pipeline that converts **English speech into Korean speech** through speech recognition, machine translation, and neural text-to-speech.

## Pipeline

```text
English Speech → Whisper → English Text → mBART-50 → Korean Text
               → g2pK/Jamo → FastSpeech2 → Mel-Spectrogram → VocGAN → Korean Speech
```

## What I Implemented

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

## Running

The pipeline requires preprocessed KSS artifacts, a FastSpeech2 checkpoint, and a pretrained VocGAN checkpoint. Large datasets and model weights are intentionally excluded.

```powershell
$env:KSS_DATA_PATH="C:/path/to/kss"
python src/pipeline.py
```

The supplied pipeline uses FastSpeech2 checkpoint step `350000`.

## Notes

This repository is a cleaned portfolio version of the original project. Generated audio, datasets, checkpoints, and pretrained vocoder weights are excluded for portability. The supplied project files did not contain quantitative STT, translation, or TTS benchmark metrics, so no unsupported metrics are claimed here.
