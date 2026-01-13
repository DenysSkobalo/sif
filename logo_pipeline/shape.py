import cv2
import numpy as np

def hu_similarity(cnt1, cnt2):
    hu1 = cv2.HuMoments(cv2.moments(cnt1)).flatten()
    hu2 = cv2.HuMoments(cv2.moments(cnt2)).flatten()

    # log-transform (CRITICAL)
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-12)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-12)

    dist = np.linalg.norm(hu1 - hu2)

    return np.exp(-dist)


def shape_similarity(cnt1, cnt2):
    score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
    return np.exp(-score)
