import cv2
import numpy as np
import random

def augment_image(img):
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # Random brightness adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
