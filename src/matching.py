import cv2

RATIO_TEST = 0.75


def ratio_test_match(des_q, des_d, method="ORB"):
    if des_q is None or des_d is None:
        return []

    if method == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn = bf.knnMatch(des_q, des_d, k=2)

    good = []
    for m, n in knn:
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    return good
