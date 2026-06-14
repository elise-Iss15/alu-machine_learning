#!/usr/bin/env python3
"""Module for calculating the summation of i squared."""


def summation_i_squared(n):
    """Calculate the sum of i^2 from i=1 to n.

    Args:
        n: the stopping condition (upper bound of summation)

    Returns:
        Integer value of the sum, or None if n is not valid.
    """
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n < 1:
        return None
    n = int(n)
    return (n * (n + 1) * (2 * n + 1)) // 6
