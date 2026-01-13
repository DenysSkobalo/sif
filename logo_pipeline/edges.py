import cv2
import numpy as np

def edge_orientation_hist(edges, bins=16):
    gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    hist = np.histogram(angle, bins=bins, range=(0, 360))[0]
    hist = hist / (hist.sum() + 1e-6)
    return hist

def orientation_similarity(h1, h2):
    return np.dot(h1, h2)

def extract_edges(gray):
    edges = cv2.Canny(gray, 80, 160)
    return edges

def extract_contours(edges):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours
