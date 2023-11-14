import numpy as np
from numpy.fft import fft2, ifft2


def gaussfft(pic, t=1.0):
    F_hat = fft2(pic)
    width, height = pic.shape
    x_range = range(int(-width / 2), int(-width / 2) + width)
    y_range = range(int(-height / 2), int(-height / 2) + height)
    x, y = np.meshgrid(x_range, y_range)

    gaussian_filter = (1 / (2 * np.pi * t)) * np.exp(-(x**2 + y**2) / (2 * t))
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

    filter_hat = np.abs(fft2(gaussian_filter))

    res = F_hat * filter_hat
    result = ifft2(res)

    return result
