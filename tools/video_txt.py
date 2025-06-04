# tools/generate_dataset_with_mediapipe.py

import cv2
import os
import random
import mediapipe as mp

# 📁 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../tools
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # .../
VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')

# YOLO 데이터셋 저장 경로
IMG_TRAIN_DIR = os.path.join(ROOT_DIR, 'datasets/images/train')
LBL_TRAIN_DIR = os.path.join(ROOT_DIR, 'datasets/labels/train')

os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
os.makedirs(LBL_TRAIN_DIR, exist_ok=True)

# 🤚 Mediapipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

print("💡 Mediapipe 기반 YOLO 데이터셋 생성 시작")

# 🎯 YOLO 라벨 포맷으로 변환
def get_yolo_bbox(img_width, img_height, x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# 🎞 비디오 반복 처리
for video_file in os.listdir(VIDEO_DIR):
    if not video_file.endswith('.mp4'):
        continue

    label_name = '_'.join(os.path.splitext(video_file)[0].split('_')[:2])
    label_index = int(label_name.split('_')[0])  # 예: '0_G' → 0

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            frame_count += 1
            continue

        # 가장 큰 손을 바운딩 박스로 계산
        image_h, image_w, _ = frame.shape
        x_list, y_list = [], []
        for landmark in results.multi_hand_landmarks[0].landmark:
            x_list.append(landmark.x * image_w)
            y_list.append(landmark.y * image_h)

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        x_center, y_center, w, h = get_yolo_bbox(image_w, image_h, x_min, y_min, x_max, y_max)

        # 파일 저장
        img_filename = f"{label_name}_{frame_count}.jpg"
        lbl_filename = f"{label_name}_{frame_count}.txt"

        cv2.imwrite(os.path.join(IMG_TRAIN_DIR, img_filename), frame)
        with open(os.path.join(LBL_TRAIN_DIR, lbl_filename), 'w') as f:
            f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        saved_count += 1
        frame_count += 1

    cap.release()
    print(f"✅ {video_file} 완료 - {saved_count}개 프레임 저장됨")

hands.close()
print("🎉 Mediapipe 데이터셋 생성 완료!")
