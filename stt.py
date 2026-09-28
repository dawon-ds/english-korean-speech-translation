import whisper

def run_stt(audio_path: str) -> str:
    """Whisper로 영어 음성을 영어 텍스트로 변환"""
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_path)
    return result["text"]
