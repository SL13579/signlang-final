# tools/generate_yolo_dataset_from_videos.py

import cv2
import random
import os
import torch

# ✅ signlang_yolo 기준 상대 경로 (현재 스크립트 위치는 tools/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../signlang_yolo/tools
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # .../signlang_yolo

VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'best.pt')

# ✅ 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True).to('cpu')

def run_yolo_prediction(frame):
    results = model(frame)
    preds = results.xyxy[0]
    if preds is None or len(preds) == 0:
        return None
    x1, y1, x2, y2 = map(int, preds[0][:4])
    return x1, y1, x2, y2

# ✅ 클래스 리스트 생성 (0_G, 1_N 등)
class_list = sorted(set([
    '_'.join(os.path.splitext(f)[0].split('_')[:2])
    for f in os.listdir(VIDEO_DIR)
    if f.endswith('.mp4')
]))
class_to_index = {name: idx for idx, name in enumerate(class_list)}

print("💡 YOLO 데이터셋 자동 생성 시작")
print(f"🧾 클래스 매핑: {class_to_index}")

# ✅ 비디오 반복 처리
for video_file in os.listdir(VIDEO_DIR):
    if not video_file.endswith('.mp4'):
        continue

    label_candidate = '_'.join(os.path.splitext(video_file)[0].split('_')[:2])
    if label_candidate not in class_to_index:
        print(f"⚠ '{video_file}' → 라벨 없음, 스킵")
        continue

    label_index = class_to_index[label_candidate]
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))

    frame_count = 0
    saved_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            bbox = run_yolo_prediction(frame)
            if bbox is None:
                print(f"⚠ 손 미탐지: {video_file} - frame {frame_count}")
                frame_count += 1
                continue

            h, w, _ = frame.shape
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # ✅ train/val 분리
            if random.random() < 0.8:
                image_dir = os.path.join(ROOT_DIR, 'datasets/images/train')
                label_dir = os.path.join(ROOT_DIR, 'datasets/labels/train')
            else:
                image_dir = os.path.join(ROOT_DIR, 'datasets/images/val')
                label_dir = os.path.join(ROOT_DIR, 'datasets/labels/val')

            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            image_name = f"{label_candidate}_{frame_count}.jpg"
            label_name = f"{label_candidate}_{frame_count}.txt"

            cv2.imwrite(os.path.join(image_dir, image_name), frame)
            with open(os.path.join(label_dir, label_name), 'w') as f:
                f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            frame_count += 1
            saved_frame_count += 1

    except KeyboardInterrupt:
        print("🛑 중단됨 (현재까지 저장된 프레임 유지)")

    cap.release()
    print(f"✅ {video_file} 처리 완료 - {saved_frame_count} 프레임 저장됨")

print("🎉 YOLO 학습용 데이터셋 자동 변환 완료!")
