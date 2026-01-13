import cv2
import numpy as np

from .edges import extract_edges, extract_contours
from .shape import hu_similarity, shape_similarity
from .sift_edges import sift_on_edges
from .score_fusion import fuse_scores
from .helpers import select_top_contours, best_shape_match, contour_complexity

def run_logo_pipeline(q_gray, dataset):
    q_edges = extract_edges(q_gray)
    q_contours_all = extract_contours(q_edges)

    q_contours = select_top_contours(q_contours_all, k=3)
    if not q_contours:
        return []

    q_complexities = [contour_complexity(c) for c in q_contours]

    sift = cv2.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(q_gray, None)
    if des_q is None:
        return []

    results = []

    for img_gray, path in dataset:
        if "flickr_logos_27_dataset" not in path:
            continue

        edges = extract_edges(img_gray)
        contours_all = extract_contours(edges)
        d_contours = select_top_contours(contours_all, k=3)
        if not d_contours:
            continue

        # contour complexity gating
        filtered_d = []
        for dc in d_contours:
            dc_comp = contour_complexity(dc)
            if any(abs(dc_comp - qc) <= 8 for qc in q_complexities):
                filtered_d.append(dc)

        if not filtered_d:
            continue

        hu, shape = best_shape_match(q_contours, filtered_d)
        shape_score = 0.6 * hu + 0.4 * shape

        if shape_score < 0.45:
            continue

        kp_d, des_d = sift.detectAndCompute(img_gray, None)
        if des_d is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(des_q, des_d, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        sift_score = min(len(good) / 50.0, 1.0)

        score = fuse_scores(
            hu_score=hu,
            shape_score=shape,
            sift_score=sift_score
        )

        if score < 0.35:
            continue

        results.append((path, score, good, kp_d))

    return sorted(results, key=lambda x: x[1], reverse=True)
