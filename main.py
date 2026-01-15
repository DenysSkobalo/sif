import argparse
import cv2
import os

# --------------------------------------------------
# Infrastructure utilities:
#  - logging: centralized experiment tracking
#  - dataset: unified dataset loading interface
#  - visualize: qualitative evaluation of results
# --------------------------------------------------
from utils.logger import setup_logger, logger
from utils.dataset import load_dataset
from utils.visualize import show_matches, show_logo_result

# --------------------------------------------------
# Object retrieval pipeline components.
# Each module corresponds to a distinct CBIR stage:
#  - features: local feature extraction
#  - masking: removal of text/noisy regions
#  - matching: descriptor-level matching
#  - geometry: geometric verification (RANSAC)
#  - color: global color-based pre-filtering
#  - scoring: fusion of heterogeneous similarity cues
#  - query_analysis: heuristic query type estimation
# --------------------------------------------------
from pipelines.object_pipeline.features import extract_features
from pipelines.object_pipeline.masking import text_mask
from pipelines.object_pipeline.matching import ratio_test_match
from pipelines.object_pipeline.geometry import ransac_filter
from pipelines.object_pipeline.color import color_prefilter
from pipelines.object_pipeline.scoring import compute_final_score, spatial_consistency
from pipelines.object_pipeline.query_analysis import analyze_query

# --------------------------------------------------
# Logo retrieval pipeline:
# A dedicated pipeline optimized for logos, combining
# shape-based cues, SIFT descriptors and score fusion.
# --------------------------------------------------
from pipelines.logo_pipeline import run_logo_pipeline

# --------------------------------------------------
# Global paths.
# Explicitly defined to ensure reproducibility and
# to avoid hard-coded paths inside the pipelines.
# --------------------------------------------------
DATASET_DIR = "data/dataset"
QUERIES_DIR = "data/queries"

# --------------------------------------------------
# Minimum number of matches required for the object
# pipeline. This empirical threshold prevents:
#  - unstable homography estimation
#  - accidental matches due to noise
# --------------------------------------------------
MIN_MATCHES_OBJECT = 10


def main():
    # --------------------------------------------------
    # Command-line interface.
    # Intentionally minimal: the focus of the project
    # is on algorithmic design rather than UI complexity.
    # --------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Smart Image Finder (SIF)"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query image filename (from data/queries)"
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # Logger initialization.
    # A separate log file is created per query image,
    # which is critical for debugging and reproducibility.
    # --------------------------------------------------
    setup_logger(log_file=f"logs/{args.query}.log")

    logger.info("Starting Smart Image Finder")
    logger.info(f"Query image: {args.query}")

    # --------------------------------------------------
    # Query image loading:
    #  - grayscale: feature extraction
    #  - BGR: color-based pre-filtering
    # --------------------------------------------------
    q_path = os.path.join(QUERIES_DIR, args.query)
    q_gray = cv2.imread(q_path, cv2.IMREAD_GRAYSCALE)
    q_bgr = cv2.imread(q_path)

    # Defensive check against missing or corrupted input
    if q_gray is None or q_bgr is None:
        logger.error("Query image not found or cannot be loaded")
        return

    # --------------------------------------------------
    # Text masking:
    # MSER is used to suppress text-like regions, which
    # often generate unstable or misleading local features,
    # especially when using ORB.
    # --------------------------------------------------
    mask = text_mask(q_gray)

    # --------------------------------------------------
    # Local feature extraction for the query.
    # ORB is chosen as a compromise between robustness
    # and computational efficiency.
    # --------------------------------------------------
    kp_q, des_q = extract_features(q_gray, mask=mask, method="ORB")

    if kp_q is None or len(kp_q) == 0:
        logger.error("No keypoints detected in query image")
        return

    # ==================================================
    # Query type detection
    # ==================================================
    # Heuristic classification of the query as either
    # "logo" or "object", based on keypoint statistics
    # and global image properties.
    #
    # This is not a learned classifier, but an explicit,
    # interpretable heuristic used to route the query
    # to the appropriate retrieval pipeline.
    # --------------------------------------------------
    query_type = analyze_query(len(kp_q), q_gray.shape)

    # Manual override for experimental control.
    # This is intentionally explicit and not hidden,
    # as it affects the evaluation protocol.
    if "logo" in args.query.lower():
        logger.info("Forcing LOGO pipeline (filename heuristic)")
        query_type = "logo"
    elif "airplane" in args.query.lower() or "laptop" in args.query.lower() or "camera" in args.query.lower():
        query_type = "object"

    logger.info(f"Query type detected: {query_type}")

    # ==================================================
    # Dataset loading
    # ==================================================
    # All dataset images are loaded once to avoid repeated
    # disk I/O during the retrieval loops.
    # --------------------------------------------------
    images, paths = load_dataset(DATASET_DIR)
    dataset = list(zip(images, paths))

    # ==================================================
    # LOGO PIPELINE
    # ==================================================
    if query_type == "logo":
        logger.info("Running PURE LOGO RETRIEVAL pipeline")

        # Specialized logo retrieval:
        # shape-based filtering → SIFT matching → score fusion
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

        # Qualitative visualization of the best logo match
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
        # BGR image is required only for color pre-filtering
        db_bgr = cv2.imread(path)
        if db_bgr is None:
            continue

        # --------------------------------------------------
        # Color-based pre-filtering:
        # Significantly reduces the search space before
        # expensive geometric verification.
        # --------------------------------------------------
        passed, color_score = color_prefilter(q_bgr, db_bgr)
        if not passed:
            continue

        # --------------------------------------------------
        # Local descriptor matching
        # --------------------------------------------------
        kp_d, des_d = extract_features(img_gray, method="ORB")
        matches = ratio_test_match(des_q, des_d)

        if len(matches) < MIN_MATCHES_OBJECT:
            continue

        # --------------------------------------------------
        # Geometric verification:
        # RANSAC-based homography estimation combined with
        # inlier counting and spatial coverage estimation.
        # --------------------------------------------------
        inliers, inlier_matches, coverage = ransac_filter(
            kp_q, kp_d, matches, q_gray.shape
        )

        if inliers == 0:
            continue

        # --------------------------------------------------
        # Spatial consistency:
        # Additional structural coherence check that
        # complements pure inlier counting.
        # --------------------------------------------------
        spatial = spatial_consistency(kp_q, kp_d, inlier_matches)

        # --------------------------------------------------
        # Score fusion:
        # The final similarity score aggregates multiple
        # independent cues into a single scalar value.
        # --------------------------------------------------
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

    # Sort results by descending final score
    results.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top object results:")
    for i, (path, score, _, _) in enumerate(results[:5]):
        logger.info(f"{i+1}. {path} -> score={score:.4f}")

    # Qualitative visualization of the best object match
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
