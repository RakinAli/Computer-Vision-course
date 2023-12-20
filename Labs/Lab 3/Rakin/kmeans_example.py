import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from Functions import mean_segments, overlay_bounds

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
        distances = np.linalg.norm(flattened_data[remaining_indices] - flattened_data[chosen_indices, None, :], axis=2)
        max_distance_index = remaining_indices[np.argmax(np.min(distances, axis=0))]
        chosen_indices.append(max_distance_index)

    # Extract the initial centers
    initial_centers = flattened_data[chosen_indices]
    return initial_centers

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
    h, w, d = image.shape
    Ivec = np.reshape(image, (h * w, d))

    # Apply k-means clustering algorithm
    # Initialize cluster centers randomly
    centers = initialize_cluster_centers(image, K, seed)

    # Initialize nearest_clustery
    nearest_cluster = np.zeros_like(Ivec[:, 0])  

    for _ in range(L):
        # Assign pixels to the nearest cluster center by using the distance to the cluster centers
        distances = distance_matrix(Ivec, centers)
        nearest_cluster = np.argmin(distances, axis=1)

        # Update cluster centers
        for k in range(K):
            if np.any(nearest_cluster == k):
                centers[k] = np.mean(Ivec[nearest_cluster == k], axis=0)
            else:
                # Reinitialize the center if the cluster is empty
                centers[k] = Ivec[np.random.choice(Ivec.shape[0])]
    
    # Reshape the segmentation and centers
    segmentation = np.reshape(nearest_cluster, (h, w))
    return segmentation, centers

def kmeans_example():
    K = 10              # number of clusters used
    L = 50              # number of iterations
    seed = 14           # seed used for random initialization
    scale_factor = 0.5  # image downscale factor
    image_sigma = 1.0   # image preblurring scale
    
    img = Image.open('Images-jpg/orange.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
    
    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    
    segm, centers = kmeans_segm(I, K, L, seed)
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img.save('result/kmeans.png')

if __name__ == '__main__':
    sys.exit(kmeans_example())
