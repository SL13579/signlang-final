# app/hand_preprocess.py
import cv2
import numpy as np

def remove_background_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
