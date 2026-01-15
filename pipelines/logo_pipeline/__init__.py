import cv2
import numpy as np

# --------------------------------------------------
# Logo pipeline primitives:
#  - edges / contours: shape extraction
#  - shape metrics: Hu moments + contour matching
#  - SIFT: local, scale-invariant descriptors
#  - score fusion: aggregation of heterogeneous cues
# --------------------------------------------------
from .edges import extract_edges, extract_contours
from .shape import hu_similarity, shape_similarity
from .sift_edges import sift_on_edges
from .score_fusion import fuse_scores
from utils.helpers import select_top_contours, best_shape_match, contour_complexity


def run_logo_pipeline(q_gray, dataset):
    """
    Execute a specialized logo retrieval pipeline.

    This pipeline is explicitly designed for logo-like queries, where:
      - the object is mostly planar,
      - shape carries strong semantic information,
      - texture may be sparse or repetitive.

    The pipeline combines:
      (1) contour-based shape filtering,
      (2) SIFT-based local feature matching,
      (3) late score fusion for robustness.
    """

    # --------------------------------------------------
    # Edge and contour extraction for the query image.
    # Edges are used as a proxy for logo shape, assuming
    # high contrast and well-defined boundaries.
    # --------------------------------------------------
    q_edges = extract_edges(q_gray)
    q_contours_all = extract_contours(q_edges)

    # --------------------------------------------------
    # Keep only the most salient contours.
    # This reduces noise and focuses computation on
    # the dominant structural elements of the logo.
    # --------------------------------------------------
    q_contours = select_top_contours(q_contours_all, k=3)
    if not q_contours:
        # No meaningful shape information available
        return []

    # --------------------------------------------------
    # Contour complexity acts as a coarse structural
    # descriptor (number of polygonal vertices).
    # It is later used for fast candidate pruning.
    # --------------------------------------------------
    q_complexities = [contour_complexity(c) for c in q_contours]

    # --------------------------------------------------
    # SIFT is chosen here for its robustness to scale
    # and rotation, which are common in logo datasets.
    # --------------------------------------------------
    sift = cv2.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(q_gray, None)
    if des_q is None:
        # Texture-less or extremely clean logos may fail here
        return []

    results = []

    # --------------------------------------------------
    # Iterate over the dataset.
    # A dataset-level filter is applied to ensure that
    # this pipeline operates only on the logo benchmark.
    # --------------------------------------------------
    for img_gray, path in dataset:
        if "flickr_logos_27_dataset" not in path:
            # Explicit separation between logo and object datasets
            continue

        # --------------------------------------------------
        # Edge and contour extraction for the database image.
        # --------------------------------------------------
        edges = extract_edges(img_gray)
        contours_all = extract_contours(edges)
        d_contours = select_top_contours(contours_all, k=3)
        if not d_contours:
            continue

        # --------------------------------------------------
        # Complexity-based contour filtering.
        # Only contours with similar structural complexity
        # to the query are retained.
        #
        # This acts as a fast, interpretable gating mechanism
        # before more expensive shape and SIFT computations.
        # --------------------------------------------------
        filtered_d = []
        for dc in d_contours:
            dc_comp = contour_complexity(dc)
            if any(abs(dc_comp - qc) <= 8 for qc in q_complexities):
                filtered_d.append(dc)

        if not filtered_d:
            continue

        # --------------------------------------------------
        # Shape similarity computation.
        # The best match over all contour pairs is retained.
        #
        # Hu moments capture global shape similarity,
        # while matchShapes captures contour alignment.
        # --------------------------------------------------
        hu, shape = best_shape_match(q_contours, filtered_d)
        shape_score = 0.6 * hu + 0.4 * shape

        # Early rejection based on shape consistency.
        # This prevents SIFT from dominating when shape
        # evidence is weak or misleading.
        if shape_score < 0.45:
            continue

        # --------------------------------------------------
        # SIFT descriptor extraction for the database image.
        # --------------------------------------------------
        kp_d, des_d = sift.detectAndCompute(img_gray, None)
        if des_d is None:
            continue

        # --------------------------------------------------
        # Descriptor matching using the classical Lowe
        # ratio test to reject ambiguous correspondences.
        # --------------------------------------------------
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(des_q, des_d, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # --------------------------------------------------
        # Normalized SIFT score.
        # The raw number of matches is capped to avoid
        # domination by very textured images.
        # --------------------------------------------------
        sift_score = min(len(good) / 50.0, 1.0)

        # --------------------------------------------------
        # Late fusion of heterogeneous similarity cues.
        # Each cue captures a different aspect of logo
        # similarity (global shape vs local texture).
        # --------------------------------------------------
        score = fuse_scores(
            hu_score=hu,
            shape_score=shape,
            sift_score=sift_score
        )

        # Final acceptance threshold.
        # Empirically tuned to balance recall and precision.
        if score < 0.35:
            continue

        results.append((path, score, good, kp_d))

    # --------------------------------------------------
    # Results are ranked by descending fused score.
    # --------------------------------------------------
    return sorted(results, key=lambda x: x[1], reverse=True)
