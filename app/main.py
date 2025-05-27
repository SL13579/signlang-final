from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.sign_detector import run_yolo_prediction
from app.stt import speech_to_text
from app.i18n import get_translations
from app.sequence_buffer import get_combined_text, clear_buffer, append_char_if_new
from app.yolo_postprocess import process_prediction
from app.hand_preprocess import remove_background_hsv
from app.hand_landmarks import get_3d_landmarks
from app.language_model import refine_sentence
from app.char_classifier import predict_char

import cv2
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/sequence")
async def get_current_sequence():
    return {"sequence": get_combined_text()}


@app.post("/sequence/clear")
async def clear_current_sequence():
    clear_buffer()
    return {"message": "버퍼가 초기화되었습니다."}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/start")


@app.get("/start", response_class=HTMLResponse)
async def start(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("start.html", {"request": request, "texts": texts})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("home.html", {"request": request, "texts": texts})


@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("camera.html", {"request": request, "texts": texts})


@app.get("/mic", response_class=HTMLResponse)
async def mic(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("mic.html", {"request": request, "texts": texts})


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("help.html", {"request": request, "texts": texts})


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    lang = request.cookies.get("lang", "ko")
    texts = get_translations(lang)
    return templates.TemplateResponse("settings.html", {"request": request, "texts": texts})

# app/main.py (수정된 predict 부분만)
@app.get("/predict")
async def predict_sign_language():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "카메라를 열 수 없습니다."}

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "카메라 오류: 프레임을 읽을 수 없습니다."}

        # YOLO 예측 실행
        result = run_yolo_prediction(frame)
        if result is None:
            return {"error": "손이 인식되지 않았습니다."}

        x1, y1, x2, y2, predicted_char = result
        cropped = frame[y1:y2, x1:x2]

        # HSV 배경 제거 → MediaPipe 좌표 → 자모 분류기
        processed_hand = remove_background_hsv(cropped)
        landmarks_vector = get_3d_landmarks(processed_hand)
        if landmarks_vector is None:
            return {"error": "손 랜드마크 인식 실패"}

        # 좌표 기반 자모 재확인
        predicted_char = predict_char(landmarks_vector)

        # 중복 제거 후 버퍼에 추가
        append_char_if_new(predicted_char)

        # 문장 보정
        sentence = refine_sentence(get_combined_text())

        return {"text": predicted_char, "sentence": sentence}

    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}

@app.get("/set_language/{lang}")
async def set_language(lang: str):
    response = RedirectResponse(url="/settings", status_code=302)
    response.set_cookie(key="lang", value=lang)
    return response


@app.post("/speak")
async def speak(text: str = Form(...)):
    return {"text": speech_to_text(text)}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)
