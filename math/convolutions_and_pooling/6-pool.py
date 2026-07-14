#!/usr/bin/env python3
"""Performs pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
            images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
            c is the number of channels in the image.
        kernel_shape: tuple of (kh, kw) containing the kernel shape for
            the pooling.
            kh is the height of the kernel.
            kw is the width of the kernel.
        stride: tuple of (sh, sw).
            sh is the stride for the height of the image.
            sw is the stride for the width of the image.
        mode: indicates the type of pooling.
            max indicates max pooling.
            avg indicates average pooling.

    Returns:
        numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    pooled = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            image_slice = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
