import cv2
import numpy as np
from utils.logger import logger

# RANSAC reprojection threshold (in pixels)
RANSAC_THRESH = 5.0

# Minimum number of matches required to attempt homography
MIN_MATCHES = 10

# Minimum number of inliers required to estimate spatial coverage
MIN_HULL_POINTS = 3


# RANSAC-based geometric verification.
# Filters matches using homography consistency and estimates spatial coverage.
def ransac_filter(kp_q, kp_d, matches, query_shape):
    if len(matches) < MIN_MATCHES:
        logger.debug("Not enough matches for RANSAC")
        return 0, [], 0.0

    # Extract matched keypoint coordinates
    pts_q = np.float32([kp_q[m.queryIdx].pt for m in matches])
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in matches])

    # Robust homography estimation
    H, mask = cv2.findHomography(
        pts_q, pts_d, cv2.RANSAC, RANSAC_THRESH
    )

    if mask is None:
        logger.debug("RANSAC failed")
        return 0, [], 0.0

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, v in zip(matches, mask) if v]
    inliers = len(inlier_matches)

    if inliers < MIN_HULL_POINTS:
        logger.debug(f"Too few inliers for coverage: {inliers}")
        return inliers, inlier_matches, 0.0

    # Estimate spatial coverage via convex hull area
    pts = pts_q[mask]
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)

    q_area = query_shape[0] * query_shape[1]
    coverage = area / q_area if q_area > 0 else 0.0

    return inliers, inlier_matches, coverage


# Measure spatial compactness of inlier matches.
# Useful as an additional structural consistency cue.
def shape_compactness(kp_d, inlier_matches):
    if len(inlier_matches) < 5:
        return 0.0

    pts = np.float32(
        [kp_d[m.trainIdx].pt for m in inlier_matches]
    )

    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)

    if perimeter == 0:
        return 0.0

    # Normalized compactness measure in [0, 1]
    compactness = 4 * np.pi * area / (perimeter ** 2)
    return min(compactness, 1.0)
