#!/usr/bin/env python3
"""Defines the MultiNormal class representing a Multivariate
Normal distribution.
"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize a MultiNormal instance.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data
                set
                n is the number of data points
                d is the number of dimensions in each data point

        Sets:
            mean: numpy.ndarray of shape (d, 1) containing the mean
                of data
            cov: numpy.ndarray of shape (d, d) containing the
                covariance matrix of data

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if n is less than 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data
                point whose PDF should be calculated
                d is the number of dimensions of the Multinomial
                instance

        Returns:
            The value of the PDF

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        x_centered = x - self.mean

        exponent = -0.5 * np.matmul(
            np.matmul(x_centered.T, inv), x_centered)
        denominator = np.sqrt(((2 * np.pi) ** d) * det)
        pdf_value = (1 / denominator) * np.exp(exponent[0][0])

        return pdf_value
