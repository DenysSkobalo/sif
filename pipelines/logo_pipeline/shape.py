import cv2
import numpy as np


# Hu-moment based shape similarity.
# Log transform is applied to stabilize dynamic range.
def hu_similarity(cnt1, cnt2):
    hu1 = cv2.HuMoments(cv2.moments(cnt1)).flatten()
    hu2 = cv2.HuMoments(cv2.moments(cnt2)).flatten()

    # Log scaling for numerical stability
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-12)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-12)

    dist = np.linalg.norm(hu1 - hu2)

    # Convert distance to similarity
    return np.exp(-dist)


# Contour similarity using OpenCV shape matching.
# Lower distance indicates better alignment.
def shape_similarity(cnt1, cnt2):
    score = cv2.matchShapes(
        cnt1, cnt2,
        cv2.CONTOURS_MATCH_I1,
        0.0
    )

    return np.exp(-score)
