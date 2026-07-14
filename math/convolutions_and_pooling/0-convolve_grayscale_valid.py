#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
            grayscale images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
            for the convolution.
            kh is the height of the kernel.
            kw is the width of the kernel.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    oh = h - kh + 1
    ow = w - kw + 1

    convolved = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            image_slice = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
