import cv2
from pipelines.logo_pipeline.shape import hu_similarity, shape_similarity


# Estimate contour complexity via polygonal approximation.
# Used as a coarse structural descriptor.
def contour_complexity(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    return len(approx)


# Select the most salient contours based on size.
# Filters out small or noisy contours.
def select_top_contours(contours, k=3, min_area=50, min_perimeter=80):
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area > min_area or peri > min_perimeter:
            valid.append(c)

    # Rank by perimeter as a proxy for visual dominance
    valid = sorted(valid, key=lambda c: cv2.arcLength(c, True), reverse=True)
    return valid[:k]


# Find the best shape correspondence between two contour sets.
# Returns the maximum similarity over all contour pairs.
def best_shape_match(q_contours, d_contours):
    best_hu = 0.0
    best_shape = 0.0

    for qc in q_contours:
        for dc in d_contours:
            hu = hu_similarity(qc, dc)
            shape = shape_similarity(qc, dc)

            best_hu = max(best_hu, hu)
            best_shape = max(best_shape, shape)

    return best_hu, best_shape
