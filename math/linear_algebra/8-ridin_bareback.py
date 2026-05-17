#!/usr/bin/env python3
"""This module performs matrix multiplication."""


def mat_mul(mat1, mat2):
    """Multiply two 2D matrices, return None if incompatible."""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            row.append(sum(mat1[i][k] * mat2[k][j]
                           for k in range(len(mat2))))
        result.append(row)
    return result
