import cv2
import numpy as np
import logging

from logger import logger   # ← КРИТИЧНО

RANSAC_REPROJ_THRESHOLD = 5.0
MIN_MATCHES = 8


def ransac_filter(kp_q, kp_d, matches):
    if len(matches) < MIN_MATCHES:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Skipping RANSAC: only {len(matches)} matches")
        return 0, []

    pts_q = np.float32([kp_q[m.queryIdx].pt for m in matches])
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(
        pts_q,
        pts_d,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD
    )

    if mask is None:
        logger.debug("RANSAC failed: no inlier mask returned")
        return 0, []

    inlier_matches = [
        m for m, inlier in zip(matches, mask.ravel()) if inlier
    ]

    return len(inlier_matches), inlier_matches
