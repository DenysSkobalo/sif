import cv2


# Apply SIFT directly on an edge map.
# Used to emphasize structural features over texture.
def sift_on_edges(edge_img):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(edge_img, None)
