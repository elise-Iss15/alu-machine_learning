#!/usr/bin/env python3
"""Calculates the intersection of obtaining data with various
hypothetical probabilities, given prior beliefs about each
probability.
"""
import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side
    effects.

    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: numpy.ndarray of shape (k,) containing the various
            hypothetical probabilities of developing severe side
            effects

    Returns:
        A 1D numpy.ndarray containing the likelihood of obtaining
        the data, x and n, for each probability in P, respectively
    """
    log_n_fact = np.sum(np.log(np.arange(1, n + 1)))
    log_x_fact = np.sum(np.log(np.arange(1, x + 1))) if x > 0 else 0
    log_nx_fact = (np.sum(np.log(np.arange(1, n - x + 1)))
                   if (n - x) > 0 else 0)
    log_n_choose_x = log_n_fact - log_x_fact - log_nx_fact

    with np.errstate(divide='ignore', invalid='ignore'):
        log_P = np.where(P > 0, np.log(P), 0.0)
        log_1mP = np.where(P < 1, np.log(1 - P), 0.0)
        log_likelihoods = log_n_choose_x + x * log_P + (n - x) * log_1mP

    likelihoods = np.exp(log_likelihoods)
    likelihoods = np.where((x == 0) & (P == 0), 1.0, likelihoods)
    likelihoods = np.where((x == n) & (P == 1), 1.0, likelihoods)
    likelihoods = np.where((x > 0) & (P == 0), 0.0, likelihoods)
    likelihoods = np.where((x < n) & (P == 1), 0.0, likelihoods)

    return likelihoods


def intersection(x, n, P, Pr):
    """Calculate the intersection of obtaining this data with the
    various hypothetical probabilities.

    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: numpy.ndarray of shape (k,) containing the various
            hypothetical probabilities of developing severe side
            effects
        Pr: numpy.ndarray of shape (k,) containing the prior
            beliefs of P

    Returns:
        A 1D numpy.ndarray containing the intersection of obtaining
        x and n with each probability in P, respectively

    Raises:
        ValueError: if n is not a positive integer
        ValueError: if x is not an integer that is greater than or
            equal to 0
        ValueError: if x is greater than n
        TypeError: if P is not a 1D numpy.ndarray
        TypeError: if Pr is not a numpy.ndarray with the same shape
            as P
        ValueError: if any value in P or Pr is not in the range
            [0, 1]
        ValueError: if Pr does not sum to 1
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    L = likelihood(x, n, P)
    intersections = L * Pr

    return intersections
