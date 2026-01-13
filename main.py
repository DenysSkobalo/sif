import argparse
import cv2
import os
import numpy as np

from utils.logger import setup_logger, logger
from src.dataset import load_dataset
from src.features import extract_features
from src.masking import text_mask
from src.matching import ratio_test_match
from src.geometry import ransac_filter
from src.visualize import show_matches
from src.color import color_prefilter
from src.scoring import compute_final_score, spatial_consistency
from src.query_analysis import analyze_query

from logo_pipeline import run_logo_pipeline

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATASET_DIR = "data/dataset"
QUERIES_DIR = "data/queries"

# --------------------------------------------------
# Thresholds (objects)
# --------------------------------------------------
MIN_MATCHES_OBJECT = 10


def show_logo_result(query_gray, db_gray, score):
    h = 600

    def resize(img):
        scale = h / img.shape[0]
        return cv2.resize(img, None, fx=scale, fy=scale)

    q = resize(query_gray)
    d = resize(db_gray)

    if q.ndim == 2:
        q = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    if d.ndim == 2:
        d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)

    vis = np.hstack([q, d])

    cv2.imshow(f"Logo match (score={score:.3f})", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Smart Image Finder (SIF)"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query image filename (from data/queries)"
    )
    args = parser.parse_args()

    setup_logger(log_file=f"logs/{args.query}.log")

    logger.info("Starting Smart Image Finder")
    logger.info(f"Query image: {args.query}")

    q_path = os.path.join(QUERIES_DIR, args.query)
    q_gray = cv2.imread(q_path, cv2.IMREAD_GRAYSCALE)
    q_bgr = cv2.imread(q_path)

    if q_gray is None or q_bgr is None:
        logger.error("Query image not found or cannot be loaded")
        return

    mask = text_mask(q_gray)
    kp_q, des_q = extract_features(q_gray, mask=mask, method="ORB")

    if kp_q is None or len(kp_q) == 0:
        logger.error("No keypoints detected in query image")
        return

    # --------------------------------------------------
    # Query type detection
    # --------------------------------------------------
    query_type = analyze_query(len(kp_q), q_gray.shape)

    if "logo" in args.query.lower():
        logger.info("Forcing LOGO pipeline (filename heuristic)")
        query_type = "logo"
    elif "airplane" in args.query.lower() or "laptop" in args.query.lower() or "camera" in args.query.lower():
        query_type = "object"
    logger.info(f"Query type detected: {query_type}")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    images, paths = load_dataset(DATASET_DIR)
    dataset = list(zip(images, paths))

    # ==================================================
    # LOGO PIPELINE
    # ==================================================
    if query_type == "logo":
        logger.info("Running PURE LOGO RETRIEVAL pipeline")

        logo_results = run_logo_pipeline(
            q_gray=q_gray,
            dataset=dataset
        )

        if not logo_results:
            logger.warning("No logo candidates found")
            return

        logger.info("Top logo results:")
        for i, (path, score, _, _) in enumerate(logo_results[:5]):
            logger.info(f"{i+1}. {path} -> score={score:.4f}")

        # visualize best logo
        best_path, best_score, _, _ = logo_results[0]
        best_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)

        show_logo_result(q_gray, best_img, best_score)
        return

    # ==================================================
    # OBJECT PIPELINE
    # ==================================================
    logger.info("Running OBJECT pipeline")

    results = []

    for img_gray, path in dataset:
        db_bgr = cv2.imread(path)
        if db_bgr is None:
            continue

        passed, color_score = color_prefilter(q_bgr, db_bgr)
        if not passed:
            continue

        kp_d, des_d = extract_features(img_gray, method="ORB")
        matches = ratio_test_match(des_q, des_d)

        if len(matches) < MIN_MATCHES_OBJECT:
            continue

        inliers, inlier_matches, coverage = ransac_filter(
            kp_q, kp_d, matches, q_gray.shape
        )

        if inliers == 0:
            continue

        spatial = spatial_consistency(kp_q, kp_d, inlier_matches)

        final_score = compute_final_score(
            inliers=inliers,
            coverage=coverage,
            color_score=color_score,
            spatial=spatial
        )

        results.append(
            (path, final_score, inlier_matches, kp_d)
        )

    if not results:
        logger.warning("No object matches found")
        return

    results.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top object results:")
    for i, (path, score, _, _) in enumerate(results[:5]):
        logger.info(f"{i+1}. {path} -> score={score:.4f}")

    best_path, _, best_matches, best_kp = results[0]
    best_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)

    show_matches(
        q_gray,
        best_img,
        kp_q,
        best_kp,
        best_matches
    )


if __name__ == "__main__":
    main()
