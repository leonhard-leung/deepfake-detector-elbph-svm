"""
elbph.py

This module implements and enhanced Local Binary Pattern Histogram (eLBPH) feature extractor
for facial images. It supports:

1. Multi-scale LBP computation across image channels.
2. Spatially-aware histograms with block weighting based on facial landmarks
3. L2 normalization and optional dimensionality reduction (PCA)
4. Integration with cached facial landmarks to speed up feature extraction

The extracted feature vectors are suitable for downstream machine learning tasks,
such as SVM-based deepfake classification.
"""

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA

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

def _spatial_histograms(lbp_maps, num_blocks=(8,8), num_bins=256, weights=None):
    """
    Split LBP maps into blocks and compute weighted histograms.

    :param lbp_maps: List of LBP maps (per scale)
    :param num_blocks: (rows, cols) number of blocks
    :param num_bins: Number of bins for histogram
    :param weights: Optional block weight matrix
    :return: Flattened histogram vector
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

def create_block_weights(landmarks, img_shape, num_blocks=(8,8)):
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

def _feature_fusion(histograms_per_channel):
    """
    Concatenates histograms across channels.

    :param histograms_per_channel: list of flattened histograms per channel
    :return: Single concatenated vector
    """
    flat = []

    for channel_histograms in histograms_per_channel:
        for scale_hist in channel_histograms:
            flat.append(scale_hist)

    return np.concatenate(flat)

def _normalize(v):
    """
    L2 normalization of a vector.
    :return:
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def _dimensionality_reduction(X, n_components=200):
    """
    Reduce dimensionality with PCA.

    :param X: 2D feature array (num_samples, num_features)
    :param n_components: Number of PCA components
    :return: Reduced 2D array
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def extract_lbp_features(images, landmarks_list=None, scales=None, num_blocks=(8,8), n_components=None):
    """
    Extract weighted multi-scale LBP features from a list of images.

    :param images: List of NumPy images (H, W, C)
    :param scales: list of (P, R) tuples
    :param num_blocks: (rows, cols) blocks for spatial histograms
    :param n_components: number of PCA components
    :return: 2D NumPy array of feature vectors
    """
    if scales is None:
        scales = [(8,1), (16,2), (24,3)]

    all_features = []

    if landmarks_list is None:
        raise ValueError("Landmarks are not provided. Use `load_landmarks` to cache landmarks.")

    for img, lm in zip(images, landmarks_list):
        channel_histograms = []

        weights = create_block_weights(lm, img.shape[:2], num_blocks)

        for channel in range(img.shape[2]):
            lbp_maps = _multi_scale_lbp(img[:,:,channel], scales=scales)
            histograms_per_scale = _spatial_histograms(lbp_maps, num_blocks=num_blocks, weights=weights)
            channel_histograms.append(histograms_per_scale)

        fused_vector = _feature_fusion(channel_histograms)
        normalized_vector = _normalize(fused_vector)

        all_features.append(normalized_vector)

    feature_matrix = np.array(all_features)

    if n_components is not None:
        feature_matrix = _dimensionality_reduction(feature_matrix, n_components)

    return feature_matrix