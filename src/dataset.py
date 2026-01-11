import os
import cv2
from logger import logger


def load_dataset(root_dir):
    dataset = []

    logger.info(f"Loading dataset from: {root_dir}")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    logger.warning(f"Failed to load image: {path}")
                    continue

                dataset.append((img, path))

    logger.info(f"Loaded {len(dataset)} images from dataset")
    return dataset
