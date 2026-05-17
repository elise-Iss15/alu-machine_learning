#!/usr/bin/env python3
"""This module performs element-wise operations on numpy.ndarrays."""


def np_elementwise(mat1, mat2):
    """Return tuple of element-wise sum, difference, product, and quotient."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
