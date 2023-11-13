import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def gaussfft(pic, t=1.0):
    # Generate a filter based on a sampled version of the Gaussian function
    width, height = pic.shape
    x_values = np.arange(-width / 2, width / 2)
    y_values = np.arange(-height / 2, height / 2)
    x, y = np.meshgrid(x_values, y_values)

    g_function = (1/2*np.pi*t)*np.exp(-(x**2+y**2)/(2*t)) 
    g_function = g_function/np.sum(g_function) # normalize!


    # Fourier transform the original image and the filter
    F_hat = fft2(pic)
    G_hat = fft2(g_function)

    # Multiply the Fourier transforms
    res = F_hat*G_hat

    # Inverse Fourier transform
    result = ifft2(res)
    

    return result
