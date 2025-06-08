# app/sign_detector.py
import torch
import cv2
import numpy as np
import os

model_path = 'yolov5/runs/train/exp4/weights/best.pt'

if os.path.exists(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
else:
    print("⚠️ YOLO 모델 파일이 존재하지 않아, 모델 로딩 생략됨")
    model = None  # 또는 예외 대신 처리 가능한 기본값

# 숫자 index → 자모 매핑 테이블
index_to_char = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ',
    5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ',
    19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅔ', 26: 'ㅒ', 27: 'ㅖ',
    28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ',
}

def run_yolo_prediction(frame):
    """
    YOLOv5를 사용해 손을 탐지하고, 바운딩박스 좌표와 자모를 반환합니다.
    """
    if model is None:
        print("⚠️ YOLO 모델이 로드되지 않았습니다. 예측을 생략합니다.")
        return None

    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        results = model(img)
        preds = results.xyxy[0]
        class_names = results.names

        if preds is None or len(preds) == 0:
            print("👋 손이 인식되지 않았습니다.")
            return None

        x1, y1, x2, y2 = preds[0][:4].cpu().numpy().astype(int)
        class_id = int(preds[0][5].item())
        class_name = class_names[class_id]  # 예: '0-1_G'

        # 하이픈 앞 숫자 추출
        try:
            index = int(class_name.split('-')[0])
            predicted_char = index_to_char.get(index, '?')
        except:
            predicted_char = '?'

        print(f"🎯 YOLO 예측: {class_name} → '{predicted_char}'")
        return x1, y1, x2, y2, predicted_char

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return None
