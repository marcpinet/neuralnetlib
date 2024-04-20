import sys
import time

import numpy as np


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encoded labels are binary vectors representing categorical values,
    with exactly one high (or "hot" = 1) bit indicating the presence of a specific category
    and all other bits low (or "cold" = 0)."""
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    labels = labels.astype(int)
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def dict_with_ndarray_to_dict_with_list(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def dict_with_list_to_dict_with_ndarray(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = np.array(v)
    return d


def apply_threshold(y_pred, threshold: float = 0.5):
    """Applies a threshold to the predictions. Typically used for binary classification."""
    return (y_pred > threshold).astype(int)


def shuffle(x, y, random_state: int = None) -> tuple:
    rng = np.random.default_rng(random_state if random_state is not None else int(time.time_ns()))
    indices = rng.permutation(len(x))
    return x[indices], y[indices]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Transform 4 dimensional images to 2 dimensional array.

    Args:
        input_data (np.ndarray): 4 dimensional input images (The number of images, The number of channels, Height, Width)
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int or tuple): the interval of stride
        pad (int or tuple): the interval of padding

    Returns:
        col (np.ndarray): 2 dimensional array

    """
    N, C, H, W = input_data.shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    # Make sure that the convolution can be executed
    assert (
                       H + 2 * pad_h - filter_h) % stride_h == 0, f'invalid parameters, (H + 2 * pad_h - filter_h) % stride_h != 0, got H={H}, pad_h={pad_h}, filter_h={filter_h}, stride_h={stride_h}'
    assert (
                       W + 2 * pad_w - filter_w) % stride_w == 0, f'invalid parameters, (W + 2 * pad_w - filter_w) % stride_w != 0, got W={W}, pad_w={pad_w}, filter_w={filter_w}, stride_w={stride_w}'

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            col[:, :, y, x, :, :] = padded_input[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Inverse of im2col.

    Args:
        col (np.ndarray): 2 dimensional array
        input_shape (tuple): the shape of original input images
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int or tuple): the interval of stride
        pad (int or tuple): the interval of padding

    Returns:
        image (np.ndarray): original images

    """
    N, C, H, W = input_shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    # Make sure that the convolution can be executed
    assert (
                       H + 2 * pad_h - filter_h) % stride_h == 0, f'invalid parameters, (H + 2 * pad_h - filter_h) % stride_h != 0, got H={H}, pad_h={pad_h}, filter_h={filter_h}, stride_h={stride_h}'
    assert (
                       W + 2 * pad_w - filter_w) % stride_w == 0, f'invalid parameters, (W + 2 * pad_w - filter_w) % stride_w != 0, got W={W}, pad_w={pad_w}, filter_w={filter_w}, stride_w={stride_w}'

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    image = np.zeros((N, C, H + 2 * pad_h + stride_h - 1, W + 2 * pad_w + stride_w - 1))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            image[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    return image[:, :, pad_h:H + pad_h, pad_w:W + pad_w]


def progress_bar(current: int, total: int, width: int = 30, message: str = "") -> None:
    progress = current / total
    bar = '=' * int(width * progress) + '-' * (width - int(width * progress))
    percent = int(100 * progress)
    sys.stdout.write(f'\r[{bar}] {percent}% {message}')
    sys.stdout.flush()
