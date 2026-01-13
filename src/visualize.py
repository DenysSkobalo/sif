import cv2


def visualize_edges(query_edges, result_edges):
    q = resize_to_height(query_edges, 400)
    r = resize_to_height(result_edges, 400)

    combined = cv2.hconcat([q, r])
    cv2.imshow("Edges (Query | Result)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_to_height(img, h=600):
    scale = h / img.shape[0]
    return cv2.resize(img, None, fx=scale, fy=scale)


def show_matches(query, db, kp_q, kp_d, matches):
    q = resize_to_height(query)
    d = resize_to_height(db)

    vis = cv2.drawMatches(
        q, kp_q, d, kp_d,
        matches[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_to_height(img, h=600):
    scale = h / img.shape[0]
    return cv2.resize(img, None, fx=scale, fy=scale)


def show_matches(query, db, kp_q, kp_d, matches):
    q = resize_to_height(query)
    d = resize_to_height(db)

    vis = cv2.drawMatches(
        q, kp_q, d, kp_d,
        matches[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
