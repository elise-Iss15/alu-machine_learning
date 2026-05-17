#!/usr/bin/env python3
"""This module concatenates two numpy.ndarrays along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy.ndarrays along a specific axis."""
    return np.concatenate((mat1, mat2), axis=axis)
