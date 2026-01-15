import cv2
import numpy as np

# Compute a normalized histogram of edge orientations.
# Used as a lightweight global shape descriptor.
def edge_orientation_hist(edges, bins=16):
    gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1)

    angle = cv2.phase(gx, gy, angleInDegrees=True)
    hist = np.histogram(angle, bins=bins, range=(0, 360))[0]

    # Normalization for scale invariance
    hist = hist / (hist.sum() + 1e-6)
    return hist


# Simple similarity between two orientation histograms.
# Acts as a coarse, global structural cue.
def orientation_similarity(h1, h2):
    return np.dot(h1, h2)


# Edge extraction using Canny.
# Assumes sufficient contrast and clean boundaries.
def extract_edges(gray):
    return cv2.Canny(gray, 80, 160)


# Extract external contours from an edge map.
# Focuses on dominant outer shapes.
def extract_contours(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours
