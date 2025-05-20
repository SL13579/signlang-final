import torch
import cv2
import numpy as np

# YOLO 모델 로딩 (손 탐지 전용)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

def run_yolo_prediction(frame):
    """
    입력 프레임에서 손의 바운딩 박스를 YOLOv5로 탐지하고, 
    가장 신뢰도 높은 바운딩 박스 좌표를 반환합니다.
    """
    try:
        # BGR → RGB 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # YOLO-friendly size

        # YOLO 추론
        results = model(img)
        preds = results.xyxy[0]
        if preds is None or len(preds) == 0:
            print("👋 손이 인식되지 않았습니다.")
            return None

        x1, y1, x2, y2 = preds[0][:4].cpu().numpy().astype(int)
        return x1, y1, x2, y2
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return None
