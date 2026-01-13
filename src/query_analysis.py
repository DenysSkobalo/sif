def analyze_query(kp_count, img_shape, contours=None):
    h, w = img_shape[:2]
    area = h * w
    density = kp_count / area

    aspect = max(h, w) / min(h, w)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        fill = cv2.contourArea(largest) / area
    else:
        fill = 0.0

    if (
        kp_count < 600 and
        density > 1e-3 and
        aspect < 2.0 and        
        fill < 0.6             
    ):
        return "logo"

    return "object"
