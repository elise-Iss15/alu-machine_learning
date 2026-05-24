#!/usr/bin/env python3
"""This module calculates the inverse of a matrix."""


def determinant(matrix):
    """Calculate and return the determinant of a matrix."""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(n):
        sub = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub)
    return det


def inverse(matrix):
    """Calculate and return the inverse of a matrix, or None if singular."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or not all(
            len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    det = determinant(matrix)
    if det == 0:
        return None
    n = len(matrix)
    if n == 1:
        return [[1 / matrix[0][0]]]
    cofactor = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [r[:j] + r[j+1:] for r in (matrix[:i] + matrix[i+1:])]
            row.append(((-1) ** (i + j)) * determinant(sub))
        cofactor.append(row)
    adjugate = [[cofactor[j][i] for j in range(n)] for i in range(n)]
    return [[adjugate[i][j] / det for j in range(n)] for i in range(n)]
