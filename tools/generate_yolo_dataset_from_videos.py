# tools/generate_yolo_dataset_from_videos.py
import cv2
import random
import sys
import os

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.sign_detector import run_yolo_prediction

# 경로 설정
VIDEO_DIR = 'videos'

# 클래스 리스트 생성 (파일명 기준)
class_list = sorted([
    os.path.splitext(file)[0]
    for file in os.listdir(VIDEO_DIR)
    if file.endswith('.mp4')
])
class_to_index = {name: idx for idx, name in enumerate(class_list)}

print("💡 비디오 → YOLO 데이터셋 생성 시작 (자동 8:2 train/val)...")
print(f"클래스 매핑: {class_to_index}")

# 영상 반복 처리
for video_file in os.listdir(VIDEO_DIR):
    if not video_file.endswith('.mp4'):
        continue

    label_candidate = os.path.splitext(video_file)[0]
    if label_candidate not in class_to_index:
        print(f"⚠ '{video_file}' → 라벨 매핑 없음, 스킵")
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

            # YOLO 입력에 맞게 크기 조정
            frame = cv2.resize(frame, (640, 640))

            # YOLOv5로 손 탐지
            bbox = run_yolo_prediction(frame)
            if bbox is None:
                print(f"⚠ 손 미탐지: {video_file} - frame {frame_count}")
                frame_count += 1
                continue

            h, w, _ = frame.shape
            x1, y1, x2, y2 = bbox

            # YOLO 형식으로 변환 (정규화)
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # 랜덤 80% → train, 20% → val
            if random.random() < 0.8:
                image_subdir = 'datasets/images/train'
                label_subdir = 'datasets/labels/train'
            else:
                image_subdir = 'datasets/images/val'
                label_subdir = 'datasets/labels/val'

            os.makedirs(image_subdir, exist_ok=True)
            os.makedirs(label_subdir, exist_ok=True)

            image_filename = f"{label_candidate}_{frame_count}.jpg"
            label_filename = f"{label_candidate}_{frame_count}.txt"

            cv2.imwrite(os.path.join(image_subdir, image_filename), frame)
            with open(os.path.join(label_subdir, label_filename), 'w') as f_label:
                f_label.write(f"{label_index} {x_center} {y_center} {width} {height}\n")

            frame_count += 1
            saved_frame_count += 1

    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨. 현재까지 저장된 데이터 유지")

    cap.release()
    print(f"✅ {video_file} 처리 완료 ({saved_frame_count} 프레임 저장됨)")

print("🎉 전체 비디오 → YOLO 데이터셋 변환 (train/val 자동 분리) 완료")
