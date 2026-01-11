import cv2


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_h))
    return resized, scale


def scale_keypoints(keypoints, scale):
    scaled = []
    for kp in keypoints:
        scaled.append(
            cv2.KeyPoint(
                x=kp.pt[0] * scale,
                y=kp.pt[1] * scale,
                size=kp.size * scale,          
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
        )
    return scaled


def show_matches_scaled(
    img_q,
    kp_q,
    img_d,
    kp_d,
    matches,
    target_height=600,
    title="Matches"
):
    img_d_resized, scale = resize_to_height(img_d, target_height)

    kp_d_scaled = scale_keypoints(kp_d, scale)

    vis = cv2.drawMatches(
        img_q, kp_q,
        img_d_resized, kp_d_scaled,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
