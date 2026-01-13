import cv2

ORB_NFEATURES = 1500


def extract_orb(image, mask=None):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    return orb.detectAndCompute(image, mask)


def extract_sift(image, mask=None):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, mask)


def extract_features(image, mask=None, method="ORB"):
    if method == "SIFT":
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(image, mask)
    else:
        orb = cv2.ORB_create(nfeatures=1500)
        return orb.detectAndCompute(image, mask)
