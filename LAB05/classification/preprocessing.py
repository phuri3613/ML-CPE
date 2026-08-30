import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    if image is None or image.size == 0:
        return None

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )
    return image

def to_features(images):
    features = images.reshape(len(images), -1).astype(np.float32)
    features /= 255.0
    return features

def preprocess_images(images, img_size=100):
    processed = [preprocess_image(img, img_size) for img in images]
    processed = [img for img in processed if img is not None]
    return to_features(np.stack(processed))