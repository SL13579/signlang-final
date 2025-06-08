# app/stt.py
import os
import wave
import json

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = None
    KaldiRecognizer = None
    print("⚠️ vosk 라이브러리를 불러올 수 없습니다.")

# 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-ko")

# 모델 초기화
if Model and os.path.exists(MODEL_PATH):
    model = Model(MODEL_PATH)
    print("✅ STT 모델 로딩 성공")
else:
    model = None
    print(f"⚠️ STT 모델 로드 생략됨. 경로 없음 또는 vosk 미설치: {MODEL_PATH}")

def speech_to_text(audio_path: str) -> str:
    """
    16kHz, 16비트, 모노 WAV 파일을 텍스트로 변환합니다.
    """
    if not model:
        print("❌ STT 모델이 로드되지 않았습니다.")
        return ""

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