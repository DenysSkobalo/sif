import cv2
import numpy as np


# Suppress text-like regions using MSER.
# Reduces unstable keypoints caused by high-contrast text.
def text_mask(gray):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    # Initialize mask: 255 = keep, 0 = suppress
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for r in regions:
        hull = cv2.convexHull(r.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 0, -1)

    return mask
