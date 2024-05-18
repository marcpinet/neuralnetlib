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


def apply_threshold(y_pred, threshold: float = 0.5):
    """Applies a threshold to the predictions. Typically used for binary classification."""
    return (y_pred > threshold).astype(int)


def im2col_2d(input_data, filter_h, filter_w, stride=1, pad=0):
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


def im2col_1d(input_data, filter_size, stride=1, pad=0):
    """
    Transform 3 dimensional images to 2 dimensional array.

    Args:
        input_data (np.ndarray): 3 dimensional input images (The number of images, The number of channels, Length)
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        col (np.ndarray): 2 dimensional array

    """
    N, C, L = input_data.shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad, pad)), mode='constant')

    col = np.zeros((N, C, filter_size, out_l))

    for y in range(filter_size):
        y_max = y + stride * out_l
        col[:, :, y, :] = padded_input[:, :, y:y_max:stride]

    col = col.transpose(0, 3, 1, 2).reshape(N * out_l, -1)
    return col


def col2im_1d(col, input_shape, filter_size, stride=1, pad=0):
    """
    Inverse of im2col_1d.

    Args:
        col (np.ndarray): 2 dimensional array
        input_shape (tuple): the shape of original input images
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        image (np.ndarray): original images

    """
    N, C, L = input_shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    col = col.reshape(N, out_l, C, filter_size).transpose(0, 2, 3, 1)

    image = np.zeros((N, C, L + 2 * pad + stride - 1))

    for y in range(filter_size):
        y_max = y + stride * out_l
        image[:, :, y:y_max:stride] += col[:, :, y, :]

    return image[:, :, pad:L + pad]


def col2im_2d(col, input_shape, filter_h, filter_w, stride=1, pad=0):
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


def pad_sequences(sequences: np.ndarray, max_length: int, padding: str = 'pre', truncating: str = 'pre') -> np.ndarray:
    """Pads sequences to the same length.

    Args:
        sequences (np.ndarray): List of sequences.
        max_length (int): Maximum length of sequences.
        padding (str): 'pre' or 'post', pad either before or after each sequence.
        truncating (str): 'pre' or 'post', remove values from sequences larger than max_length, either at the beginning or at the end of the sequences.

    Returns:
        np.ndarray: Padded sequences.
    """
    padded_sequences = np.zeros((len(sequences), max_length))
    for i, sequence in enumerate(sequences):
        if len(sequence) > max_length:
            if truncating == 'pre':
                sequence = sequence[-max_length:]
            else:
                sequence = sequence[:max_length]
        if padding == 'pre':
            padded_sequences[i, -len(sequence):] = sequence
        else:
            padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_

    def transform(self, X):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * self.scale_ + self.min_


class PCA:
    def __init__(self, n_components: int, random_state: int = None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None
        self.mean = None
        self.input_shape = None

    def fit(self, X: np.ndarray):
        self.input_shape = X.shape[1:]
        X = X.reshape(X.shape[0], -1)

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        X_centered = X - self.mean

        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X_reconstructed = np.dot(X, self.components.T) + self.mean
        return X_reconstructed.reshape((-1, *self.input_shape))
