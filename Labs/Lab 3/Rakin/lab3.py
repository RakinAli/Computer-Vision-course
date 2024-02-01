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
import random


def openImage(path):
    """
    Implement file reading to load an image from disk.:
    - Input: path : image path input
    - Output: img : image as numpy array
    """
    img = Image.open(path)
    img = np.asarray(img).astype(np.float32)
    return img


def create_centroids(image, K, channels=3):
    # Get the unique RGB values
    values = image.reshape(-1, channels)
    selected_values = set()

    while len(selected_values) < K:
        values_picked = values[np.random.choice(values.shape[0], K)]
        # Check if the value is already in the set of selected values
        for value in values_picked:
            value = tuple(value)
            if value not in selected_values:
                # Add the value to the set of selected values
                selected_values.add(value)
    initial_centers = np.array(list(selected_values)[:K])
    return initial_centers

# Does not work properly
def create_centroids_logically(image, K, channels=3):
    # Get the unique RGB values
    values = image.reshape(-1, channels)
    selected_values = set()
    # Pick one point randomly
    values_picked = values[np.random.choice(values.shape[0], 1)]
    # Add the value to the set of selected values
    selected_values.add(tuple(values_picked[0]))
    while len(selected_values) < K:
        # Calculate the distance between each point and the already picked points
        dist = distance_matrix(values, np.array(list(selected_values)))
        # Get the maximum distance for each point
        max_dist = np.max(dist, axis=1)
        # Sort the indices of max_dist in descending order of distance
        sorted_indices = np.argsort(max_dist)[::-1]

        # Find the next unique point
        for idx in sorted_indices:
            candidate = tuple(values[idx])
            if candidate not in selected_values:
                selected_values.add(candidate)
                break

        # Add a check to handle the situation where all unique points are exhausted
        if len(selected_values) == len(np.unique(values, axis=0)):
            print("All unique points have been added.")
            break
    initial_centers = np.array(list(selected_values)[:K])
    return initial_centers


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
    np.random.seed(seed)
    random.seed(seed)
    dimensions = image.ndim

    # Check if the image is already flat or if it's 3D
    if dimensions == 3:  # If image is 3D
        height, width, channels = image.shape
        image_flat = np.reshape(image, (-1, 3))

    # If image is already 2D
    elif dimensions == 2:
        height, channels = image.shape
        image_flat = np.reshape(image, (-1, channels))
    else:
        raise ValueError("Image must be a 2D or 3D array.")

    # K-means algorithm
    centers = create_centroids(image, K)

    # Iterate L times
    for i in range(L):
        # Calculate distance matrix
        dist = distance_matrix(image_flat, centers)

        # Assign each pixel to closest cluster
        segmentation = np.argmin(dist, axis=1)

        # Update cluster centers
        for j in range(K):
            centers[j] = np.mean(image_flat[segmentation == j], axis=0)

    # Reshape segmentation to original image dimensions
    if dimensions == 3:
        segmentation = segmentation.reshape(height, width)

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

    """K-means segmentation - HERE IS the error """
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
