import numpy as np
from scipy.signal import convolve2d, correlate2d
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from PIL import Image
from Functions import *
from gaussfft import gaussfft


def openImage(path):
    """
    Implement file reading to load an image from disk.:
    - Input: path : image path input
    - Output: img : image as numpy array
    """
    img = Image.open(path)
    img = np.asarray(img).astype(np.float32)
    return img

def kmeans_segm(image, K, L, seed):
    """
    Implement k-means clustering:
    - Input: image : image input
             K : number of cluster centres
             L : number of iterations
             seed : to initialize random generator
    - Output: segmentation : segmented image with a colour index per pixel
              centers : final cluster centres in 3D colour space
    """
    # This is used to flatten the image
    Ivec = np.reshape(image, (-1, 3))

    # Apply k-means clustering algorithm
    # Initialize cluster centers randomly
    centers = Ivec[np.random.seed(seed), :]

    # Initialize nearest_clustery
    nearest_cluster = np.zeros_like(Ivec[:, 0])  

    for _ in range(L):
        # Assign pixels to the nearest cluster center by using pyhtagoras equation
        distances = np.sqrt(((Ivec[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        nearest_cluster = np.argmin(distances, axis=1)

        # Update cluster centers
        for k in range(K):
            centers[k] = Ivec[nearest_cluster == k].mean(axis=0)

    segmentation = np.reshape(nearest_cluster, (-1, 3))
    return segmentation, centers

def mixture_prob(image, K, L, mask):
    # Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).
        # Store all pixels for which mask=1 in a Nx3 matrix
        # Randomly initialize the K components using masked pixels
        # Iterate L times
        # Expectation: Compute probabilities P_ik using masked pixels
        # Maximization: Update weights, means and covariances using masked pixels
        # Compute probabilities p(c_i) in Eq.(3) for all pixels I.
    
    # Convert input image into numpy array and normalize it
    image = np.asarray(image) / 255

    # Reshape into 2d array where is row is a pixels RGB value
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)

    # Reshape mask into 1d array
    mask = np.reshape(mask, (-1))
    masked_Ivec = Ivec[np.reshape(np.nonzero(mask == 1), (-1))]

    # Get Gaussian components.
    segmentation, centers = kmeans_segm(masked_Ivec, K, L, 35)

    # Create Covariance matrix all init to identity
    cov = np.zeros((K, 3, 3))
    for i in range(K):
        cov[i] = np.eye(3) * 0.01

    # Initializes weights (weights) based on the proportion of pixels
    weights = np.zeros(K)
    for i in range(K):
        weights[i] = np.sum(np.nonzero(segmentation == i)).astype(int) / segmentation.shape[0]

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
