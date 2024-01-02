import sys
import math
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from Functions import showgrey, mean_segments, overlay_bounds
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix


def initialize_cluster_centers(image, K, seed):
    np.random.seed(seed)
    h, w, d = image.shape
    flattened_data = image.reshape(h * w, d)

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
            axis=2,
        )
        max_distance_index = remaining_indices[np.argmax(np.min(distances, axis=0))]
        chosen_indices.append(max_distance_index)

    # Extract the initial centers
    initial_centers = flattened_data[chosen_indices]
    return initial_centers

def kmeans_segm(image, K, L, seed=42):
    """
    Input Args:
        Image - the RBG input image
        K - the number of clusters
        L - the number of iterations
        seed - the initialization seed
    Output:
        Segmentation - Integer image with cluster indices
        centers - an array with K cluster mean colors
    """
    height, width, dimensions = image.shape
    # Flatten the image
    image_flat = np.reshape(image, (height * width, dimensions))

    # K-means algorithm
    # Initialize the cluster centers randomly
    centers = initialize_cluster_centers(image, K, seed)

    # Iterate L times
    for i in range(L):
        # Assign each pixel to the cluster center for which the distance is minimum
        distances = distance_matrix(image_flat, centers)
        nearest_cluster = np.argmin(distances, axis=1)

        # Update cluster centers
        for j in range(K):
            if np.any(nearest_cluster == j):
                centers[j, :] = np.mean(image_flat[nearest_cluster == j], axis=0)
            else:
                centers[j] = image_flat[np.random.choice(image_flat.shape[0])]

    # Reshape the segmentation and centers
    segmentation = np.reshape(nearest_cluster, (height, width))
    return segmentation, centers

def mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations):
    print("Find colour channels with K-means...")
    K = 16  # number of channels
    [segm, centers] = kmeans_segm(I, K, 20, 4321)
    (height, width, depth) = np.shape(I)
    idx = np.reshape(segm, (height, width))
    mapsw = np.zeros((height, width, K))
    mapsx = np.zeros((height, width, K))
    mapsy = np.zeros((height, width, K))
    [X, Y] = np.meshgrid(range(width), range(height))
    for k in range(K):
        mapsw[:, :, k] = (idx == k).astype(float)
        mapsx[:, :, k] = gaussian_filter(
            mapsw[:, :, k] * X, spatial_bandwidth, mode="nearest"
        )
        mapsy[:, :, k] = gaussian_filter(
            mapsw[:, :, k] * Y, spatial_bandwidth, mode="nearest"
        )
        mapsw[:, :, k] = gaussian_filter(
            mapsw[:, :, k], spatial_bandwidth, mode="nearest"
        )
    mapsw = np.reshape(mapsw, (-1, K)) + 1e-6
    mapsx = np.reshape(mapsx, (-1, K))
    mapsy = np.reshape(mapsy, (-1, K))

    print("Search for high density points...")
    constC = -0.5 / (colour_bandwidth**2)
    x = np.reshape(X, (width * height,))
    y = np.reshape(Y, (width * height,))
    Ic = np.reshape(I, (width * height, 3))
    wei = np.exp(constC * (distance_matrix(Ic, centers) ** 2))
    for l in range(num_iterations):
        p = (np.round(y) * width + np.round(x)).astype(int)
        ww = mapsw[p, :] * wei
        w = np.sum(ww, axis=1)
        u = (np.matmul(ww, centers).T / w).T
        x = ((np.sum(mapsx[p, :] * wei, axis=1)).T / w).T
        y = ((np.sum(mapsy[p, :] * wei, axis=1)).T / w).T
        wei = (ww.T / w).T
        x = np.maximum(np.minimum(x, width - 1), 0)
        y = np.maximum(np.minimum(y, height - 1), 0)

    print("Assign high density points to pixels...")
    XY = np.stack((x, y))
    thr = 4.0
    val = 0
    mask = np.zeros((height * width, 1), dtype=np.short)
    for y in range(height):
        for x in range(width):
            p = y * width + x
            if mask[p] == 0:
                stack = [p]
                val = val + 1
                mask[p] = val
                while len(stack) > 0:
                    p0 = stack[-1]
                    xy = XY[:, p0]
                    y0 = int(p0 / width)
                    x0 = p0 - y0 * width
                    stack = stack[:-1]
                    pn = p0 + 1
                    if (
                        x0 < width - 1
                        and mask[pn] == 0
                        and (np.sum((xy - XY[:, pn]) ** 2) < thr)
                    ):
                        stack = stack + [pn]
                        mask[pn] = val
                    pn = p0 - 1
                    if (
                        x0 > 0
                        and mask[pn] == 0
                        and (np.sum((xy - XY[:, pn]) ** 2) < thr)
                    ):
                        stack = stack + [pn]
                        mask[pn] = val
                    pn = p0 + width
                    if (
                        y0 < height - 1
                        and mask[pn] == 0
                        and (np.sum((xy - XY[:, pn]) ** 2) < thr)
                    ):
                        stack = stack + [pn]
                        mask[pn] = val
                    pn = p0 - width
                    if (
                        y0 > 0
                        and mask[pn] == 0
                        and (np.sum((xy - XY[:, pn]) ** 2) < thr)
                    ):
                        stack = stack + [pn]
                        mask[pn] = val
    segm = np.reshape(mask, (height, width))
    return segm


def mean_shift_example():
    scale_factor = 0.5  # image downscale factor
    image_sigma = 1.0  # image preblurring scale
    spatial_bandwidth = 10.0  # spatial bandwidth
    colour_bandwidth = 20.0  # colour bandwidth
    num_iterations = 40  # number of mean-shift iterations

    img = Image.open("Images-jpg/tiger1.jpg")
    img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)))

    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)

    segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations)
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img.save("result/meanshift.png")


if __name__ == "__main__":
    sys.exit(mean_shift_example())
