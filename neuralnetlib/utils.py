import sys
import time
import os
import platform
import subprocess
import numpy as np

class History(dict):
    """A custom dictionary that doesn't display its contents when returned in Jupyter."""
    def __repr__(self):
        return ""


def dict_with_ndarray_to_dict_with_list(d: dict) -> dict:
    """Converts all numpy arrays in a dictionary to lists. This is useful for serializing the dictionary to JSON."""
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def dict_with_list_to_dict_with_ndarray(d: dict) -> dict:
    """Converts all lists in a dictionary to numpy arrays. This is useful for deserializing the dictionary from JSON."""
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = np.array(v)
    return d


def shuffle(x: np.ndarray, y: np.ndarray, random_state: int = None) -> tuple:
    """Shuffles the data along the first axis."""
    x = np.array(x)
    y = np.array(y)
    rng = np.random.default_rng(random_state if random_state is not None else int(time.time_ns()))
    indices = rng.permutation(len(x))
    return x[indices], y[indices]


def progress_bar(current: int, total: int, width: int = 30, message: str = "") -> None:
    """
    Prints a progress bar to the console.
    
    Args:
        current (int): current progress
        total (int): total progress
        width (int): width of the progress bar
        message (str): message to display next to the progress bar
    """
    progress = current / total
    bar = '=' * int(width * progress) + '-' * (width - int(width * progress))
    percent = int(100 * progress)
    sys.stdout.write(f'\r[{bar}] {percent}% {message}')
    sys.stdout.flush()


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> tuple:
    """
    Splits the data into training and test sets.

    Args:
        x (np.ndarray or list): input data
        y (np.ndarray or list): target data
        test_size (float): the proportion of the dataset to include in the test split
        random_state (int): seed for the random number generator

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    x = np.array(x)
    y = np.array(y)
    rng = np.random.default_rng(random_state if random_state is not None else int(time.time_ns()))
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split_index = int(len(x) * (1 - test_size))
    x_train = x[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_train = y[indices[:split_index]]
    y_test = y[indices[split_index:]]
    return x_train, x_test, y_train, y_test


def is_interactive():
    try:
        import __main__ as main
        return not hasattr(main, '__file__')
    except:
        return False


def is_display_available():
    system = platform.system()
    
    if system == "Linux":
        return is_display_available_linux()
    elif system == "Windows":
        return is_display_available_windows()
    else:
        raise NotImplementedError(f"Display check not implemented for {system}")


def is_display_available_linux():
    if "DISPLAY" in os.environ:
        try:
            output = subprocess.check_output(["xdpyinfo"], stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError:
            return False
    return False


def is_display_available_windows():
    try:
        import win32api
        display_devices = win32api.EnumDisplayDevices()
        return bool(display_devices)
    except Exception:
        return False