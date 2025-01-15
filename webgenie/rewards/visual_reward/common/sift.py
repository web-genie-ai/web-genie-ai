import numpy as np
from skimage import io, color
from skimage.feature import SIFT
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def extract_sift_from_roi(gray_image, roi):
    # ROI: (x, y, w, h)
    x, y, w, h = roi
    # Crop the image to that sub-region
    roi_image = gray_image[y : y + h, x : x + w]

    # Initialize SIFT
    sift = SIFT()
    sift.detect_and_extract(roi_image)

    # SIFT keypoints are relative to the top-left of the ROI
    keypoints = sift.keypoints
    descriptors = sift.descriptors

    return keypoints, descriptors

def match_sift_features(kp1, desc1, kp2, desc2, distance_metric="euclidean", threshold=0.75):
    if (desc1 is None or len(desc1) == 0) and (desc2 is None or len(desc2) == 0):
        return 1
    elif (desc1 is None or len(desc1) == 0) or (desc2 is None or len(desc2) == 0):
        return 0

    # Compute the pairwise distance matrix between descriptors
    cost_matrix = cdist(desc1, desc2, metric=distance_metric)

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Apply a threshold to filter out matches with large distances
    matches = []
    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] < threshold:
            matches.append((i, j))

    return 1 - len(matches) / max(len(kp1), len(kp2))
