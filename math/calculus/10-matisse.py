#!/usr/bin/env python3
"""Module for calculating the derivative of a polynomial."""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial.

    Args:
        poly: list of coefficients where index = power of x.
              e.g. [5, 3, 0, 1] represents 5 + 3x + 0x^2 + x^3

    Returns:
        New list of coefficients of the derivative,
        [0] if derivative is 0, or None if poly is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [i * poly[i] for i in range(1, len(poly))]
    if all(c == 0 for c in derivative):
        return [0]
    return derivative
