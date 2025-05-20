# app/hand_landmarks.py
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def get_3d_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return np.array(landmark_list).flatten()
    return None