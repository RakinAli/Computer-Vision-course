from matplotlib.figure import Figure
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from pick import pick

from Functions import *
from fftwave import fftwave


def question1():
    points = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]

    for point in points:
        fftwave(point[0], point[1])


def main():
    print("Which section of the lab are you interested in?")
    print("1. Section 1: Properties of discrte Fourier transform")
    print("2. Section 2: Gaussian convolution via FFT")
    print("3. Section 3: Smoothing ")
    print("4. All of the sections")

    choice = input("Enter your choice: Between number 1 to 4: ")

    if choice == "1":
        question1()

    elif choice == "2":
        print("Not implemented yet")

    elif choice == "3":
        print("Not implemented yet")

    elif choice == "4":
        print("Not implemented yet")

    else:
        print("Invalid choice")
        exit(1)


main()
