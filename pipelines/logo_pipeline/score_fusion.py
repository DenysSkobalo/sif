# Linearly combine heterogeneous similarity scores.
# Weights reflect the relative trust in each cue.
def fuse_scores(hu_score, shape_score, sift_score,
                w_hu=0.4, w_shape=0.3, w_sift=0.3):

    return (
        w_hu * hu_score +
        w_shape * shape_score +
        w_sift * sift_score
    )
