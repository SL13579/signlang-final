# app/stt.py
import os
import wave
import json
from vosk import Model, KaldiRecognizer

# 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-ko")

# 모델 초기화
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"STT 모델 경로가 존재하지 않습니다: {MODEL_PATH}")

model = Model(MODEL_PATH)

def speech_to_text(audio_path: str) -> str:
    """
    16kHz, 16비트, 모노 WAV 파일을 텍스트로 변환합니다.
    """
    with wave.open(audio_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("WAV 파일은 16kHz, 모노, 16비트 형식이어야 합니다.")

        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)

        result = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if recognizer.AcceptWaveform(data):
                partial = json.loads(recognizer.Result())
                result += partial.get("text", "") + " "

        final = json.loads(recognizer.FinalResult())
        result += final.get("text", "")
        return result.strip()
