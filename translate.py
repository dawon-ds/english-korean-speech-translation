import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Translation model using: {device}")

tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
).to(device)

def translate_en_to_ko(text: str) -> str:
    """영어 → 한국어 번역 (mBART 기반)"""
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def run_translate(text: str) -> str:
    return translate_en_to_ko(text)
