import numpy as np
from scipy.signal import convolve2d, correlate2d
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