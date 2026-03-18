import cv2
import numpy as np
import random

def augment_image(img, augment_type='random'):
    """
    Apply controlled augmentation to an image.

    Parameters:
    -----------
    img : numpy array (BGR)
        Input image
    augment_type : str
        'random' - applies random augmentation
        'flip' - horizontal flip only
        'rotate' - small rotation (±15°)
        'brightness' - brightness/contrast adjustment
        'zoom' - minor zoom/crop

    Returns:
    --------
    Augmented image (BGR numpy array)
    """
    if augment_type == 'random':
        augment_type = random.choice(['flip', 'rotate', 'brightness', 'zoom'])

    h, w = img.shape[:2]

    if augment_type == 'flip':
        # Horizontal flip only (vertical flips are unrealistic for snakes)
        img = cv2.flip(img, 1)

    elif augment_type == 'rotate':
        # Small rotation (±15 degrees)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif augment_type == 'brightness':
        # Brightness/contrast adjustment in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Adjust V (brightness) channel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
        # Slight saturation adjustment
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.9, 1.1), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif augment_type == 'zoom':
        # Minor zoom/crop (95-105%)
        scale = random.uniform(0.95, 1.05)
        new_w, new_h = int(w * scale), int(h * scale)

        if scale > 1.0:
            # Zoom in: resize larger, then crop center
            resized = cv2.resize(img, (new_w, new_h))
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # Zoom out: resize smaller, then pad
            resized = cv2.resize(img, (new_w, new_h))
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = cv2.copyMakeBorder(resized, pad_y, h - new_h - pad_y,
                                      pad_x, w - new_w - pad_x,
                                      cv2.BORDER_REFLECT)

    return img


def generate_augmented_images(img, num_augments=3):
    """
    Generate multiple augmented versions of an image.

    Parameters:
    -----------
    img : numpy array (BGR)
        Input image
    num_augments : int
        Number of augmented images to generate

    Returns:
    --------
    List of augmented images
    """
    augmented = []
    aug_types = ['flip', 'rotate', 'brightness', 'zoom']

    for i in range(num_augments):
        # Use different augmentation type for variety
        aug_type = aug_types[i % len(aug_types)]
        aug_img = augment_image(img.copy(), augment_type=aug_type)
        augmented.append(aug_img)

    return augmented
