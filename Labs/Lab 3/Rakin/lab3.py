import numpy as np
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt
from PIL import Image
from Functions import *
from gaussfft import gaussfft
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import distance


def initialize_cluster_centers(image, K, seed):
    np.random.seed(seed)

    # Check if the image is already flat or if it's 3D
    if image.ndim == 3:  # If image is 3D
        h, w, d = image.shape
        flattened_data = image.reshape(h * w, d)
    elif image.ndim == 2:  # If image is already 2D
        flattened_data = image
        h, w = flattened_data.shape[0], 1  # Treat the image as having a single column
        d = flattened_data.shape[1]
    else:
        raise ValueError("Image must be a 2D or 3D array.")

    # Initialize the list of chosen indices
    chosen_indices = []

    # Randomly pick the first center
    first_index = np.random.choice(h * w)
    chosen_indices.append(first_index)

    # Iteratively pick the most different pixels
    for _ in range(1, K):
        remaining_indices = [i for i in range(h * w) if i not in chosen_indices]
        distances = np.linalg.norm(
            flattened_data[remaining_indices] - flattened_data[chosen_indices, None, :],
            axis=2 if image.ndim == 3 else 1,
        )
        max_distance_index = remaining_indices[np.argmax(np.min(distances, axis=0))]
        chosen_indices.append(max_distance_index)

    # Extract the initial centers
    initial_centers = flattened_data[chosen_indices]
    return initial_centers


def openImage(path):
    """
    Implement file reading to load an image from disk.:
    - Input: path : image path input
    - Output: img : image as numpy array
    """
    img = Image.open(path)
    img = np.asarray(img).astype(np.float32)
    return img


def kmeans_segm(image, K, L, seed=42):
    """
    Input Args:
        Image - the RGB input image which can be 2D or 3D.
        K - the number of clusters
        L - the number of iterations
        seed - the initialization seed
    Output:
        Segmentation - Integer image with cluster indices
        centers - an array with K cluster mean colors
    """
    # Check if the image is already a 2D array or a 3D array
    if image.ndim == 3:
        height, width, dimensions = image.shape
        image_flat = np.reshape(image, (height * width, dimensions))
    elif image.ndim == 2:
        # The image is already flat, use it directly
        image_flat = image
        # Since the image is flat, we don't have the original height and width
        height, width = None, None
    else:
        raise ValueError("Image must be a 2D or 3D array.")

    # K-means algorithm
    # Initialize the cluster centers randomly
    centers = initialize_cluster_centers(image_flat, K, seed)

    # Iterate L times
    for i in range(L):
        # Assign each pixel to the cluster center for which the distance is minimum
        distances = distance.cdist(image_flat, centers)
        nearest_cluster = np.argmin(distances, axis=1)

        # Update cluster centers
        for j in range(K):
            if np.any(nearest_cluster == j):
                centers[j, :] = np.mean(image_flat[nearest_cluster == j], axis=0)
            else:
                # If a cluster has no pixels, reinitialize its center
                centers[j] = image_flat[np.random.choice(image_flat.shape[0])]

    if height is not None and width is not None:
        # Reshape the segmentation back to the original image's shape
        segmentation = np.reshape(nearest_cluster, (height, width))
    else:
        # If the image was originally 2D, keep the segmentation flat
        segmentation = nearest_cluster

    return segmentation, centers


def mixture_prob(image, K, L, mask):
    """Preprocessing stage for EM algorithm."""
    # Convert input image into numpy array and normalize it
    image = np.asarray(image) / 255

    # Reshape into 2d array where is row is a pixels RGB value
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)

    # Reshape mask into 1d array
    mask = np.reshape(mask, (-1))
    masked_Ivec = Ivec[np.reshape(np.nonzero(mask == 1), (-1))]

    """K-means segmentaiton"""
    # Get Gaussian components.
    segmentation, centers = kmeans_segm(masked_Ivec, K, L)

    """Initialize parameters"""
    # Create Covariance matrix all init to identity
    cov = np.zeros((K, 3, 3))
    for i in range(K):
        cov[i] = np.eye(3) * 0.01

    # Initializes weights (weights) based on the proportion of pixels
    weights = np.zeros(K)
    for i in range(K):
        weights[i] = np.sum(np.nonzero(segmentation == i)) / segmentation.shape[0]

    """Propability computation"""
    # Iteratre L times to refine the paramters of the Gaussian components
    for i in range(L):
        # Expectation step
        P_ik = np.zeros((Ivec.shape[0], K))
        for k in range(K):
            P_ik[:, k] = weights[k] * multivariate_normal(centers[k], cov[k]).pdf(Ivec)

        # Normalize P_ik
        for j in range(K):
            P_ik[:, j] = np.divide(
                P_ik[:, j], np.sum(P_ik, axis=1), where=np.sum(P_ik, axis=1) != 0
            )

        # Maximization step
        for k in range(K):
            # Update weights
            weights[k] = np.mean(P_ik[:, k])

            # Update centers
            centers[k] = np.sum(P_ik[:, k].reshape(-1, 1) * Ivec, axis=0) / np.sum(
                P_ik[:, k]
            )

            # Update covariance matrices
            cov[k] = np.sum(
                P_ik[:, k].reshape(-1, 1, 1)
                * (Ivec - centers[k]).reshape(-1, 1, 3)
                * (Ivec - centers[k]).reshape(-1, 3, 1),
                axis=0,
            ) / np.sum(P_ik[:, k])

    # Compute probabilities
    prob = np.zeros((Ivec.shape[0], K))
    for k in range(K):
        prob[:, k] = weights[k] * multivariate_normal(centers[k], cov[k]).pdf(Ivec)
        prob[:, k] = prob[:, k] / np.sum(prob[:, k])

    prob = np.sum(prob, axis=1)
    prob = np.reshape(prob, (image.shape[0], image.shape[1]))
    return prob
