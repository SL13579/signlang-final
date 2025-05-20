# tools/collect_landmarks.py
import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 데이터 저장 리스트
X_data = []
y_data = []

# videos 폴더의 mp4 기준 클래스 리스트 생성 (영어 표기)
VIDEO_DIR = 'videos'
class_list = sorted([
    os.path.splitext(f)[0].lower() for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')
])
class_to_index = {name: idx for idx, name in enumerate(class_list)}

print(f"💡 클래스 매핑 (videos 기준): {class_to_index}")
print("💡 수어 자모 좌표 수집 시작 (q 키로 종료)")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 카메라 오류")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 21개의 (x, y, z) 좌표 추출 → flatten (63)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # 프레임 표시
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Enter label (e.g., g, n, d...)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Hand Landmark Collection", frame)

            # 라벨 입력 대기 (한 번만)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

            # 라벨 입력 받기
            label_input = input(f"라벨 입력 (videos 기준 영어 표기: {class_list}): ").strip().lower()
            if label_input not in class_to_index:
                print("❗ 유효하지 않은 라벨, 무시")
                continue

            X_data.append(landmarks)
            y_data.append(class_to_index[label_input])
            print(f"✅ 저장 완료: {label_input}")

    cv2.imshow("Hand Landmark Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 데이터 저장
os.makedirs("data", exist_ok=True)
np.save("data/X_landmarks.npy", np.array(X_data))
np.save("data/y_labels.npy", np.array(y_data))

print("🎉 좌표 데이터 저장 완료: data/X_landmarks.npy, data/y_labels.npy")
