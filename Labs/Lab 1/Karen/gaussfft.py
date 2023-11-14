import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    # Generate the filter based on a sampled version of the Gausian function.
    width, height = pic.shape
    
    gridx = np.arange(-width // 2, width // 2)
    gridy = np.arange(-height // 2, height // 2)

    x, y = np.meshgrid(gridx, gridy)
    gaussian_filter = (1/(2*np.pi*t)) * np.exp(-(x**2 + y**2) / (2 * t))

    # This is used to nprmalise the filter.
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    
    # Fourier transform of the original image and the Gaussian filter.
    Fhat = fft2(pic)
    Ghat = fft2(gaussian_filter)

    # Multiply the Fourier transforms.
    FGhat = Fhat * Ghat

    # Invert the resulting Fourier transforms.
    result = ifft2(FGhat)
    
    return result
