# Convolutions and Pooling

This project implements image convolution and pooling operations from
scratch using only `numpy`, without relying on `np.convolve` or any deep
learning framework.

## Learning Objectives

By the end of this project you should be able to explain, without the
help of Google:

- What a convolution is
- What max pooling and average pooling are
- What a kernel/filter is
- What padding is
- What "same" padding and "valid" padding are
- What a stride is
- What channels are
- How to perform a convolution over an image
- How to perform max/average pooling over an image

## Requirements

- Editors: `vi`, `vim`, `emacs`
- Ubuntu 16.04 LTS, `python3` (3.5), `numpy` (1.15)
- All files start with `#!/usr/bin/env python3`
- All files end with a new line
- Code follows `pycodestyle` (2.5)
- Every module, class, and function is documented
- Only `numpy` (as `np`) and `from math import ceil, floor` may be
  imported unless otherwise stated
- `np.convolve` may not be used
- All files are executable

## Files

| File                                 | Description                                              |
| ------------------------------------ | ---------------------------------------------------------|
| `0-convolve_grayscale_valid.py`      | Valid convolution on grayscale images                    |
| `1-convolve_grayscale_same.py`       | Same convolution on grayscale images                     |
| `2-convolve_grayscale_padding.py`    | Convolution on grayscale images with custom padding      |
| `3-convolve_grayscale.py`            | Convolution on grayscale images with padding and stride  |
| `4-convolve_channels.py`             | Convolution on images with multiple channels             |
| `5-convolve.py`                      | Convolution on images using multiple kernels             |
| `6-pool.py`                          | Max/average pooling on images                             |

## Usage

Each file can be imported and used directly, e.g.:

```python
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
```

See the `*-main.py` test files in the project description for example
usage with the MNIST and animal image datasets.
