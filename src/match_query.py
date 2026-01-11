import os
import cv2
import argparse
from operator import itemgetter


DATASET_DIR = "data/dataset"
QUERIES_DIR = "data/queries"
ORB_NFEATURES = 1000
TOP_K = 5
RATIO_TEST = 0.75


def load_images_with_paths(root_dir):
    images = []
    paths = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                paths.append(path)

    return images, paths


def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_descriptors(des_q, des_d):
    if des_q is None or des_d is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_q, des_d, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return len(good_matches)


def main():
    parser = argparse.ArgumentParser(description="Smart Image Finder - Query Matching")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query image filename (from data/queries/)"
    )
    args = parser.parse_args()

    query_path = os.path.join(QUERIES_DIR, args.query)

    if not os.path.exists(query_path):
        print("[ERROR] Query image not found.")
        return

    # Load query
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        print("[ERROR] Failed to load query image.")
        return

    kp_q, des_q = extract_orb(query_img)

    if des_q is None:
        print("[WARN] No descriptors found in query image.")
        return

    print(f"[INFO] Query loaded: {args.query}")
    print("[INFO] Extracting features and matching...")

    # Load dataset
    dataset_images, dataset_paths = load_images_with_paths(DATASET_DIR)

    results = []

    for img, path in zip(dataset_images, dataset_paths):
        kp_d, des_d = extract_orb(img)
        score = match_descriptors(des_q, des_d)

        if score > 0:
            results.append((path, score))

    # Sort by score descending
    results.sort(key=itemgetter(1), reverse=True)

    print("\nTop results:")
    for i, (path, score) in enumerate(results[:TOP_K]):
        print(f"{i+1}. {path} -> score = {score}")

    # Optional visualization of best match
    if results:
        best_path, _ = results[0]
        best_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)
        kp_b, des_b = extract_orb(best_img)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des_q, des_b)

        img_match = cv2.drawMatches(
            query_img, kp_q,
            best_img, kp_b,
            matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Best match", img_match)
        print("\n[INFO] Press 'q' or ESC to close the window")

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
