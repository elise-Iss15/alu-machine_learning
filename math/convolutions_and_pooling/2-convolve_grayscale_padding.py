#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

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
        padding: tuple of (ph, pw).
            ph is the padding for the height of the image.
            pw is the padding for the width of the image.

    The image should be padded with 0's.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0,
    )

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    convolved = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            image_slice = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
