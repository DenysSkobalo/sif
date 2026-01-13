import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_SIM_THRESHOLD = 0.4


def compute_hsv_hist(image_bgr, bins=(32, 32, 32)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=bins,
        ranges=[0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist


def color_similarity(hist_q, hist_d):
    return cv2.compareHist(hist_q, hist_d, cv2.HISTCMP_CORREL)


def color_prefilter(img_q_bgr, img_d_bgr):
    hist_q = compute_hsv_hist(img_q_bgr)
    hist_d = compute_hsv_hist(img_d_bgr)

    sim = color_similarity(hist_q, hist_d)

    logger.debug(f"Color similarity = {sim:.3f}")

    return sim >= COLOR_SIM_THRESHOLD, sim
