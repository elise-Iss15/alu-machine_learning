#!/usr/bin/env python3
"""This module returns the transpose of a 2D matrix."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix as a new matrix."""
    return [[matrix[r][c] for r in range(len(matrix))]
            for c in range(len(matrix[0]))]
