import cv2


# Heuristic query type classification.
# Routes the query to either the logo or object pipeline.
def analyze_query(kp_count, img_shape, contours=None):
    h, w = img_shape[:2]
    area = h * w

    # Keypoint density as a coarse texture indicator
    density = kp_count / area

    # Aspect ratio to reject highly elongated images
    aspect = max(h, w) / min(h, w)

    # Optional fill ratio based on dominant contour
    if contours:
        largest = max(contours, key=cv2.contourArea)
        fill = cv2.contourArea(largest) / area
    else:
        fill = 0.0

    # Empirical rules favoring logo-like structure
    if (
        kp_count < 600 and
        density > 1e-3 and
        aspect < 2.0 and
        fill < 0.6
    ):
        return "logo"

    return "object"
