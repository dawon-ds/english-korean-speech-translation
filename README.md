# English-to-Korean Speech Translation Pipeline

An end-to-end speech translation pipeline that converts **English speech into Korean speech** by connecting speech recognition, machine translation, and neural text-to-speech components.

## Pipeline

```text
English speech
    ↓
Whisper (STT)
    ↓
English text
    ↓
mBART-50 (EN → KO)
    ↓
Korean text
    ↓
g2pK + Hangul/Jamo preprocessing
    ↓
FastSpeech2
    ↓
Mel-spectrogram
    ↓
VocGAN
    ↓
Korean speech
```

## What I Implemented

- Integrated Whisper-based English speech recognition into the pipeline.
- Added English-to-Korean translation with `facebook/mbart-large-50-many-to-many-mmt`.
- Connected Korean grapheme-to-phoneme preprocessing to the FastSpeech2 input sequence.
- Integrated FastSpeech2 synthesis and a pretrained VocGAN vocoder into one inference flow.
- Added CPU/GPU device selection for the STT, translation, and TTS stages.
- Reorganized local paths so dataset locations can be configured through `KSS_DATA_PATH`.

## Main Components

| Stage | Component |
| --- | --- |
| Speech-to-Text | Whisper `base.en` |
| Translation | mBART-50 many-to-many multilingual MT |
| Korean text preprocessing | g2pK, Jamo decomposition |
| Text-to-Speech | FastSpeech2 |
| Vocoder | VocGAN |
| Framework | PyTorch |

## Project Structure

```text
.
├── pipeline.py              # End-to-end inference pipeline
├── stt.py                   # Whisper STT
├── translate.py             # mBART EN→KO translation
├── synthesize.py            # FastSpeech2 synthesis
├── fastspeech2.py           # FastSpeech2 model
├── modules.py               # Variance adaptor and model modules
├── transformer/             # Transformer blocks
├── vocoder/                 # VocGAN generator
├── text/                    # Korean text processing
├── audio/                   # Audio and spectrogram utilities
├── data/                    # Dataset utilities
├── hparams.py               # Model/data configuration
├── train.py                 # FastSpeech2 training script
└── requirements.txt
```

## Running the Pipeline

The pipeline requires FastSpeech2/KSS preprocessing artifacts, a FastSpeech2 checkpoint, and a pretrained VocGAN checkpoint. Large model artifacts and datasets are intentionally excluded from this repository.

Set the KSS dataset location if needed:

```bash
# Windows PowerShell
$env:KSS_DATA_PATH="C:/path/to/kss"
```

Then place the required checkpoints in the paths configured in `hparams.py` and run:

```bash
python pipeline.py
```

The source pipeline currently uses `input_voice.wav` as the input and FastSpeech2 checkpoint step `350000`.

## Notes

This repository is organized as a portfolio version of the project. Dataset files, checkpoints, pretrained vocoder weights, and generated audio are not committed because of their size and portability. The included source demonstrates the end-to-end integration logic; no quantitative STT, translation, or TTS benchmark metrics were included in the supplied project files, so none are claimed here.
