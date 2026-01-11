import cv2

ORB_NFEATURES = 1000


def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des
