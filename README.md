# English–Korean Speech Translation Pipeline

**English speech → Korean speech**, integrating speech recognition, translation, and neural speech synthesis in a single inference workflow.

## Project Overview

A personal project focused on connecting pretrained models through compatible text, pronunciation, and audio representations.

## Approach

### Pipeline

```text
English Speech → Whisper → English Text
              ↓
          mBART-50 → Korean Text
              ↓
   g2pK / Jamo → FastSpeech2
              ↓
  Mel-Spectrogram → VocGAN → Korean Speech
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

### Training and Evaluation

FastSpeech2 optimizes five loss components: **mel**, **mel-postnet**, **duration**, **F0 (pitch)**, and **energy**. These track spectral reconstruction, speech timing, intonation, and intensity.

- `src/training/train.py` records **total training loss** to TensorBoard.
- `src/training/evaluate.py` computes the five individual validation losses and prints their values. The supplied portfolio also contains a historical TensorBoard screenshot, but the current public code does not establish the provenance of every curve in that image.

These loss values assess acoustic prediction objectives; they are not direct measurements of speech naturalness or translation accuracy.

## Results

### Speech Translation Sample

| Stage | Example |
| --- | --- |
| English speech input | “She looked into the mirror and saw another person.” |
| Korean text output | “그녀는 거울을 들여다보고 다른 사람을 봤습니다.” |

**Audio Samples**

<table>
  <tr>
    <td align="center"><a href="results/input_voice.wav">Input English speech</a></td>
    <td align="center"><a href="results/output_voice.wav">Generated Korean speech</a></td>
  </tr>
</table>

### Synthesized Spectrogram

FastSpeech2 predicts a mel-spectrogram representing frequency-band energy across time, which VocGAN converts into the final audio waveform. The synthesis model also predicts **F0** (pitch) and **energy** (speech intensity). However, the public `utils.plot_data()` implementation displays **only the mel-spectrogram**, not separate F0 or energy curves.

This demonstrates an end-to-end inference example, **not** an objective guarantee of pronunciation accuracy, perceptual audio quality, or translation performance. Quantitative STT, translation, TTS, and latency metrics are not available.

## Repository Structure

```text
.
├── src/
│   ├── pipeline.py
│   ├── inference/          # STT, translation, and synthesis
│   │   ├── stt.py
│   │   ├── translate.py
│   │   └── synthesize.py
│   ├── models/             # FastSpeech2 and vocoder
│   │   ├── fastspeech2.py
│   │   ├── transformer/
│   │   └── vocoder/
│   ├── training/           # Training and evaluation
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── preprocessing/      # KSS, text, and audio processing
│   │   ├── data/
│   │   ├── text/
│   │   └── audio/
│   ├── hparams.py
│   └── utils.py
├── results/
│   ├── input_voice.wav
│   └── output_voice.wav
├── requirements.txt
└── README.md
```

## Usage

Run from the repository root. Install dependencies and provide the required model checkpoints and KSS preprocessing artifacts (not included in the repository).

```bash
pip install -r requirements.txt
```

| Required file | Expected location |
| --- | --- |
| Input WAV | `results/input_voice.wav` |
| FastSpeech2 checkpoint | `ckpt/kss/checkpoint_350000.pth.tar` |
| KSS statistics | `preprocessed/kss/{mel,f0,energy}_stat.npy` |
| VocGAN checkpoint | `vocoder/pretrained_models/vocgan_kss_pretrained_model_epoch_4500.pt` |

**Run inference (PowerShell):**

```powershell
$env:KSS_DATA_PATH="C:/path/to/kss"
python -m src.pipeline
```

**Run inference (Bash):**

```bash
export KSS_DATA_PATH="/path/to/kss"
python -m src.pipeline
```

`KSS_DATA_PATH` specifies the source KSS dataset, not checkpoint paths. Change artifact locations in `src/hparams.py` if needed. The entry point reads `results/input_voice.wav`.

The synthesis code writes a VocGAN WAV (`_vocgan_output_voi.wav`), a Griffin–Lim WAV (`_griffin_lim_output_voi.wav`), and a mel-spectrogram image (`_output_voi.png`) to `results/`. The uploaded `output_voice.wav` is the curated demonstration sample; the generated filenames are controlled by `src/inference/synthesize.py`.

## Limitations

Whisper transcription errors may propagate into mBART translations and the final Korean speech output. The integration was demonstrated qualitatively, but the individual stages have not been evaluated with comparable, controlled accuracy or listening tests.

Reproducing the full pipeline from a fresh checkout requires external FastSpeech2/VocGAN weights and KSS preprocessing artifacts. Future work includes stage-wise evaluation to identify error sources, plus translation-quality and perceptual speech-quality measurements.

## Data and Third-party Resources

- **Third-party models:** Whisper, mBART-50, FastSpeech2 and VocGAN. Check the license and permitted uses of each particular codebase, checkpoint and dataset; trained weights and KSS audio are not redistributed.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
