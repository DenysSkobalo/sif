import os
import cv2
from utils.logger import logger

def load_dataset(root):
    images, paths = [], []

    logger.info(f"Loading dataset from: {root}")

    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                p = os.path.join(r, f)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                paths.append(p)

    logger.info(f"Loaded {len(images)} images from dataset")
    return images, paths
