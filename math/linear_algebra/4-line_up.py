#!/usr/bin/env python3
"""This module adds two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise, return None if different shapes."""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
