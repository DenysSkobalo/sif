import cv2

ORB_NFEATURES = 1500
import cv2

# Default number of ORB features.
# Chosen as a trade-off between coverage and speed.
ORB_NFEATURES = 1500


# ORB feature extraction wrapper.
def extract_orb(image, mask=None):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    return orb.detectAndCompute(image, mask)


# SIFT feature extraction wrapper.
def extract_sift(image, mask=None):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, mask)


# Unified feature extraction interface.
# Allows switching between ORB and SIFT without
# changing downstream pipeline code.
def extract_features(image, mask=None, method="ORB"):
    if method == "SIFT":
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(image, mask)
    else:
        orb = cv2.ORB_create(nfeatures=1500)
        return orb.detectAndCompute(image, mask)
