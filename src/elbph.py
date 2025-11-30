"""
elbph.py

This module implements and enhanced Local Binary Pattern Histogram (eLBPH) feature extractor
for facial images. It supports:

1. Multi-scale LBP computation across a luminance channel (Y).
2. Spatially-aware histograms with block weighting based on facial landmarks
3. L2 normalization
4. Integration with cached facial landmarks to speed up feature extraction

The extracted feature vectors are suitable for downstream machine learning tasks,
such as SVM-based deepfake classification.
"""

import numpy as np
from skimage.feature import local_binary_pattern

def _multi_scale_lbp(img, scales=None):
    """
    Compute LBP maps for multiple scales on a single-channel image.

    :param img: Single-channel image
    :param scales: List of (P, R) tuples, where P=neighbors, and R=radius
    :return: List of LBP maps per scale
    """
    if scales is None:
        scales = [(8,1), (16,2), (24,3)]

    lbp_maps = []
    for P, R in scales:
        lbp = local_binary_pattern(img, P, R, method='uniform')
        lbp_maps.append(lbp)

    return lbp_maps

def _spatial_histograms(lbp_maps, num_blocks=(6,6), num_bins=128, weights=None):
    """
    Split LBP maps into blocks and compute weighted histograms.

    :param lbp_maps: List of LBP maps (per scale)
    :param num_blocks: (rows, cols) number of blocks
    :param num_bins: Number of bins for histogram
    :param weights: Optional block weight matrix
    :return: Flattened histogram vector per scale
    """
    rows, cols = num_blocks
    height, width = lbp_maps[0].shape
    block_height, block_width = height // rows, width // cols

    histogram_list = []
    for lbp in lbp_maps:
        histograms = []
        for i in range(rows):
            for j in range(cols):
                block = lbp[
                    i*block_height:(i+1)*block_height,
                    j*block_width:(j+1)*block_width
                ]
                histogram, _ = np.histogram(block.ravel(), bins=num_bins, range=(0, num_bins))
                if weights is not None:
                    histogram = histogram * weights[i, j]
                histograms.append(histogram)
        histogram_list.append(np.concatenate(histograms))

    return histogram_list

def _feature_fusion(histograms_per_scale):
    """
    Concatenates histograms across scales.

    :param histograms_per_scale: list of flattened histograms per scale
    :return: Single concatenated vector
    """
    return np.concatenate(histograms_per_scale)

def _normalize(v):
    """
    L2 normalization of a vector.

    :return: L2-normalized vector, if the input vector has zero norm, return the original vector
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def _create_block_weights(landmarks, img_shape, num_blocks=(8,8)):
    """
    Create a weight matrix for blocks based on landmarks (eyes, mouth).

    :param landmarks: Dictionary of keypoints
    :param img_shape: Image shape (H, W, C)
    :param num_blocks: (rows, cols) number of blocks
    :return: NumPy array of shape (rows, cols)
    """
    height, width = img_shape[:2]
    rows, cols = num_blocks
    block_height, block_width = height // rows, width // cols

    weights = np.ones((rows, cols), dtype=float)
    important_points = ["left_eye", "right_eye", "mouth_left", "mouth_right"]

    for key in important_points:
        if key not in landmarks:
            continue
        x, y = landmarks[key]
        row = int(y // block_height)
        col = int(x // block_width)

        if 0 <= row < rows and 0 <= col < cols:
            weights[row, col] = 1.5

    return weights

def extract_lbp_features(images, scales=None, num_blocks=(8,8), landmarks_list=None):
    """
    Extract weighted multi-scale LBP features from a list of images.

    :param images: List of NumPy images (H, W, C)
    :param scales: list of (P, R) tuples
    :param num_blocks: (rows, cols) blocks for spatial histograms
    :param landmarks_list: List of landmarks (eyes, mouth)
    :return: 2D NumPy array of feature vectors
    """
    all_features = []

    for img, lm in zip(images, landmarks_list):
        # compute weights
        weights = _create_block_weights(lm, img.shape, num_blocks)

        # multi-scale LBP maps
        lbp_maps = _multi_scale_lbp(img=img, scales=scales)

        # spatial histograms
        histograms_per_scale = _spatial_histograms(
            lbp_maps=lbp_maps,
            num_blocks=num_blocks,
            num_bins=256,
            weights=weights
        )

        # feature fusion
        fused_vector = _feature_fusion(histograms_per_scale)

        # normalization
        normalized_vector = _normalize(fused_vector)

        all_features.append(normalized_vector)

    return np.array(all_features)


def extract_single_lbp_features(img, landmarks, scales=None, num_blocks=(8, 8), pca_model=None):
    """
    Extract weighted multi-scale LBP features from a single image (Y channel).

    :param img: NumPy image (H, W, C)
    :param landmarks: Dictionary of facial landmarks for the image
    :param scales: list of (P, R) tuples
    :param num_blocks: (rows, cols) blocks for spatial histograms
    :param pca_model: trained PCA object (optional) to reduce feature dimensionality
    :return: 1D NumPy array of feature vector
    """
    # compute weights
    weights = _create_block_weights(landmarks, img.shape[:2], num_blocks)

    # multi-scale LBP maps
    lbp_maps = _multi_scale_lbp(img, scales=scales)

    # spatial histograms
    histograms_per_scale = _spatial_histograms(lbp_maps, num_blocks=num_blocks, weights=weights)

    # feature fusion
    fused_vector = _feature_fusion(histograms_per_scale)

    # normalization
    normalized_vector = _normalize(fused_vector)

    # dimensionality reduction: optional
    if pca_model is not None:
        normalized_vector = pca_model.transform(normalized_vector.reshape(1, -1)).flatten()

    return normalized_vector