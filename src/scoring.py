import numpy as np
import math


def spatial_consistency(kp_q, kp_d, matches):
    if len(matches) < 3:
        return 0.0

    angles = []
    for m in matches:
        qx, qy = kp_q[m.queryIdx].pt
        dx, dy = kp_d[m.trainIdx].pt
        angle = math.atan2(dy - qy, dx - qx)
        angles.append(angle)

    std = np.std(angles)
    return max(0.0, 1.0 - std)


def compute_final_score(inliers, coverage, color_score, spatial,
                        w_inliers=0.4,
                        w_coverage=0.3,
                        w_color=0.15,
                        w_spatial=0.15,
                        shape=0.0,
                        w_shape=0.15):

    inliers_n = min(inliers / 50.0, 1.0)
    color_n = (color_score + 1.0) / 2.0

    return (
        w_inliers * inliers_n +
        w_coverage * coverage +
        w_color * color_n +
        w_spatial * spatial +
        w_shape * shape
    )

def class_bonus(query_label, db_label, bonus=0.25):
    if query_label is None or db_label is None:
        return 0.0
    return bonus if query_label == db_label else -bonus
