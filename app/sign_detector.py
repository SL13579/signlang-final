import torch
import cv2
import json
from app.sign_language import label_to_text
from app.sequence_buffer import append_char

# 모델 로딩
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

def run_yolo_prediction(frame):
    results = model(frame)
    predictions = results.pred[0]

    # 예외 처리 + 로그 출력
    if predictions is None or len(predictions) == 0:
        print("👋 손이 인식되지 않았습니다.")  # ✅ 콘솔에 로그 출력
        return "손이 인식되지 않았습니다."

    class_id = int(predictions[0][5].item())  # 첫 번째 감지된 클래스
    predicted_char = label_to_text(class_id)
    print(f"🔤 예측된 클래스: {class_id} → {predicted_char}")  # ✅ 예측 결과 로그
    append_char(predicted_char)

    return predicted_char
