import cv2

RATIO_TEST = 0.75


def ratio_test_match(des_q, des_d):
    if des_q is None or des_d is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des_q, des_d, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return good_matches
