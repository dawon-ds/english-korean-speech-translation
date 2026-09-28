import os
import torch
import numpy as np

from stt import run_stt
from translate import run_translate
from synthesize import get_FastSpeech2, synthesize
from text import text_to_sequence
import utils
import hparams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from g2pk import G2p
from jamo import h2j
import re

g2p = G2p()

def text_preprocess(text):
    phone = g2p(text)
    phone = h2j(phone)
    phone = list(filter(lambda p: p != " ", phone))
    phone = "{" + "}{".join(phone) + "}"
    phone = re.sub(r"\{[^\w\s]?\}", "{sil}", phone)
    phone = phone.replace("}{", " ")
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    return torch.LongTensor(sequence).unsqueeze(0).to(device)

def pipeline(input_wav_path, step=350000):
    print("\n=== [1] STT: 영어 음성 → 영어 텍스트 ===")
    english_text = run_stt(input_wav_path)
    print("STT 결과:", english_text)

    print("\n=== [2] 번역: 영어 텍스트 → 한국어 텍스트 ===")
    korean_text = run_translate(english_text)
    print("번역 결과:", korean_text)

    print("\n=== [3] TTS: 한국어 텍스트 → 한국어 음성 ===")
    model = get_FastSpeech2(step).to(device)
    vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    text_tensor = text_preprocess(korean_text)

    output_name = "output_voice"
    synthesize(
        model=model,
        vocoder=vocoder,
        text=text_tensor,
        sentence=output_name,
        dur_pitch_energy_aug=[1.0, 1.0, 1.0],
        prefix="",
    )

if __name__ == "__main__":
    pipeline("input_voice.wav", step=350000)
