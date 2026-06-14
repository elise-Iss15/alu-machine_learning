#!/usr/bin/env python3
"""Module for calculating the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial.

    Args:
        poly: list of coefficients where index = power of x.
              e.g. [5, 3, 0, 1] represents 5 + 3x + 0x^2 + x^3
        C: integer representing the integration constant (default 0)

    Returns:
        New list of coefficients of the integral (as small as possible),
        or None if poly or C are not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        power = i + 1
        result = coef / power
        if result == int(result):
            result = int(result)
        integral.append(result)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
