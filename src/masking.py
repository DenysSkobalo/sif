import cv2
import numpy as np

def text_mask(gray):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for r in regions:
        hull = cv2.convexHull(r.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 0, -1)

    return mask
