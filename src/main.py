import argparse
import os
import cv2

from features import extract_orb
from matching import ratio_test_match
from geometry import ransac_filter
from visualize import show_matches_scaled
from logger import setup_logger, logger
from dataset import load_dataset


DATASET_DIR = "data/dataset"
QUERIES_DIR = "data/queries"
TOP_K = 5


def main():
    parser = argparse.ArgumentParser(description="Smart Image Finder (SIF)")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query image filename (from data/queries)"
    )
    args = parser.parse_args()

    # ✅ ТЕПЕР args існує
    os.makedirs("logs", exist_ok=True)
    setup_logger(log_file=f"logs/{args.query}.log")

    logger.info("Starting Smart Image Finder")
    logger.info(f"Query image: {args.query}")

    query_path = os.path.join(QUERIES_DIR, args.query)
    if not os.path.exists(query_path):
        logger.error("Query image not found")
        return

    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        logger.error("Failed to load query image")
        return

    kp_q, des_q = extract_orb(query_img)
    if des_q is None:
        logger.warning("No descriptors found in query image")
        return

    dataset = load_dataset(DATASET_DIR)

    results = []

    for img, path in dataset:
        kp_d, des_d = extract_orb(img)
        matches = ratio_test_match(des_q, des_d)

        if len(matches) < 8:
            logger.debug("Not enough matches for RANSAC")
            continue

        inliers, inlier_matches = ransac_filter(kp_q, kp_d, matches)

        if inliers > 0:
            results.append((path, inliers, kp_d, matches))

    results.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top results:")
    for i, (path, score, _, _) in enumerate(results[:TOP_K]):
        logger.info(f"{i+1}. {path} -> inliers = {score}")

    if results:
        best_path, _, kp_d, matches = results[0]
        best_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)
        show_matches_scaled(
            query_img,
            kp_q,
            best_img,
            kp_d,
            matches,
            title="Best match (RANSAC inliers)"
        )


if __name__ == "__main__":
    main()
