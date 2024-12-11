import os
import platform
import subprocess
import sys
import time
import shutil

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


def shuffle(x: np.ndarray, y: np.ndarray = None, random_state: int = None) -> tuple:
    """Shuffles the data along the first axis."""
    rng = np.random.default_rng(
        random_state if random_state is not None else int(time.time_ns()))

    n_samples = x.shape[0]
    indices = rng.permutation(n_samples)

    x = np.array(x)
    shuffled_x = x[indices]

    if y is not None:
        y = np.array(y)
        shuffled_y = y[indices]
        return shuffled_x, shuffled_y

    return shuffled_x


def balanced_batch_sampling(n_classes: int, real_samples: np.ndarray, labels: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Generates a balanced batch of samples by selecting a fixed number of samples from each class.

    Args:
        n_classes (int): The number of classes
        real_samples (np.ndarray): The real samples
        labels (np.ndarray): The labels of the samples in one-hot encoding
        batch_size (int): The total number of samples to select
        rng (np.random.Generator): The random number generator

    Raises:
        ValueError: If the batch size is less than the number of classes

    Returns:
        tuple: A tuple of (real_samples, labels) where each array has the selected samples
    """
    samples_per_class = batch_size // n_classes
    if samples_per_class == 0:
        raise ValueError(f"batch_size ({batch_size}) doit être au moins égal au nombre de classes ({n_classes})")
    
    selected_indices = []
    
    class_indices = [np.nonzero(labels[:, class_idx] == 1)[0] for class_idx in range(n_classes)]
    
    empty_classes = [i for i, indices in enumerate(class_indices) if len(indices) == 0]
    if empty_classes:
        raise ValueError(f"Les classes {empty_classes} n'ont aucun échantillon dans le dataset")
    
    for class_idx in range(n_classes):
        selected_class_indices = rng.choice(
            class_indices[class_idx],
            size=samples_per_class,
            replace=True
        )
        selected_indices.extend(selected_class_indices)
    
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    
    return real_samples[selected_indices], labels[selected_indices]


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
    sys.stdout.write('\r' + ' ' * shutil.get_terminal_size().columns)
    sys.stdout.write(f'\r[{bar}] {percent}% {message}')
    sys.stdout.flush()


def train_test_split(x: np.ndarray, y: np.ndarray = None, test_size: float = 0.2, random_state: int = None, shuffle: bool = True) -> tuple:
    """
    Splits the data into training and test sets.

    Args:
        x (np.ndarray or list): input data
        y (np.ndarray or list, optional): target data. If None, only x will be split
        test_size (float): the proportion of the dataset to include in the test split
        random_state (int): seed for the random number generator
        shuffle (bool): whether to shuffle the data before splitting

    Returns:
        tuple: (x_train, x_test) if y is None, else (x_train, x_test, y_train, y_test)
    """
    x = np.array(x)
    rng = np.random.default_rng(
        random_state if random_state is not None else int(time.time_ns()))
    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)
    split_index = int(len(x) * (1 - test_size))
    x_train = x[indices[:split_index]]
    x_test = x[indices[split_index:]]

    if y is None:
        return x_train, x_test

    y = np.array(y)
    y_train = y[indices[:split_index]]
    y_test = y[indices[split_index:]]
    return x_train, x_test, y_train, y_test


def make_blobs(n_samples=100, 
              n_features=2, 
              centers=2, 
              cluster_std=1.0, 
              center_box=(-10.0, 10.0), 
              random_state=None):
    """
    Generate isotropic Gaussian blobs for clustering.
    
    Args:
        n_samples (int): The total number of points to generate
        n_features (int): The number of features for each sample
        centers (int or array): The number of centers or array of center locations
        cluster_std (float or array): The standard deviation of the clusters
        center_box (tuple): The bounding box for each center when centers are randomly generated
        random_state (int): Determines random number generation for dataset creation
    
    Returns:
        tuple: (X, y) where X is the array of samples and y is the array of integer labels
    """
    rng = np.random.default_rng(random_state)
    
    if isinstance(centers, int):
        n_centers = centers
        centers = rng.uniform(center_box[0], center_box[1], 
                            size=(n_centers, n_features))
    else:
        centers = np.array(centers)
        n_centers = centers.shape[0]
    
    if np.isscalar(cluster_std):
        cluster_std = np.array([cluster_std] * n_centers)
    
    samples_per_center = np.full(n_centers, n_samples // n_centers, dtype=int)
    samples_per_center[:n_samples - sum(samples_per_center)] += 1
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    current_pos = 0
    for i, (n_samples_center, std, center) in enumerate(
        zip(samples_per_center, cluster_std, centers)):
        
        X[current_pos:current_pos + n_samples_center] = (
            rng.normal(0, std, (n_samples_center, n_features)) + center
        )
        
        y[current_pos:current_pos + n_samples_center] = i
        
        current_pos += n_samples_center
        
    shuffle_idx = rng.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def make_classification(n_samples=100,
                       n_features=20,
                       n_informative=2,
                       n_redundant=2,
                       n_repeated=0,
                       n_classes=2,
                       n_clusters_per_class=2,
                       weights=None,
                       flip_y=0.01,
                       class_sep=1.0,
                       hypercube=True,
                       shift=0.0,
                       scale=1.0,
                       shuffle=True,
                       random_state=None):
    """
    Generate a random n-class classification problem.
    
    Args:
        n_samples: int, default=100: The number of samples
        n_features: int, default=20: The total number of features
        n_informative: int, default=2: The number of informative features
        n_redundant: int, default=2: The number of redundant features
        n_repeated: int, default=0: The number of duplicated features
        n_classes: int, default=2: The number of classes
        n_clusters_per_class: int, default=2: The number of clusters per class
        weights: list-like of float, default=None: The proportions of samples assigned to each class
        flip_y: float, default=0.01: The fraction of samples whose class is randomly flipped
        class_sep: float, default=1.0: The factor multiplying the hypercube size
        hypercube: bool, default=True: If True, generate clusters in corners of hypercube
        shift: float, default=0.0: Shift feature means by a random value from [-shift, shift]
        scale: float, default=1.0: Scale features by a random value from [1-scale, 1+scale]
        shuffle: bool, default=True: Shuffle samples and features
        random_state: int, default=None: Random state for reproducibility

    Returns:
        X: array of shape [n_samples, n_features]: The generated samples
        y: array of shape [n_samples]: The integer labels for class membership of each sample
    """
    
    rng = np.random.default_rng(random_state)
    
    n_features_total = n_informative + n_redundant + n_repeated
    if n_features_total > n_features:
        raise ValueError("Sum of informative, redundant and repeated features must be <= n_features")
    
    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()
    
    centers = []
    if hypercube:
        for i in range(n_classes):
            for j in range(n_clusters_per_class):
                center = rng.standard_normal(size=n_informative)
                center = np.sign(center) * class_sep
                centers.append(center)
    else:
        for i in range(n_classes * n_clusters_per_class):
            center = rng.standard_normal(size=n_informative) * class_sep
            centers.append(center)
    
    centers = np.array(centers)
    
    n_samples_per_cluster = []
    
    remaining_samples = n_samples
    for k in range(n_classes - 1):
        n_samples_k = int(n_samples * weights[k])
        remaining_samples -= n_samples_k
        samples_per_clusters_k = [n_samples_k // n_clusters_per_class] * n_clusters_per_class
        extra = n_samples_k % n_clusters_per_class
        for i in range(extra):
            samples_per_clusters_k[i] += 1
        n_samples_per_cluster.extend(samples_per_clusters_k)
    
    samples_per_clusters_last = [remaining_samples // n_clusters_per_class] * n_clusters_per_class
    extra = remaining_samples % n_clusters_per_class
    for i in range(extra):
        samples_per_clusters_last[i] += 1
    n_samples_per_cluster.extend(samples_per_clusters_last)
    
    X = []
    y = []
    
    for i, n in enumerate(n_samples_per_cluster):
        if n > 0:
            X.append(centers[i] + rng.standard_normal(size=(n, n_informative)))
            y.extend([i // n_clusters_per_class] * n)
    
    X = np.vstack(X)
    y = np.array(y)
    
    if n_redundant > 0:
        B = rng.standard_normal(size=(n_informative, n_redundant))
        X_redundant = np.dot(X[:, :n_informative], B)
        X = np.hstack([X, X_redundant])
    
    if n_repeated > 0:
        indices = rng.integers(0, n_informative + n_redundant, size=n_repeated)
        X_repeated = X[:, indices]
        X = np.hstack([X, X_repeated])
    
    n_noise = n_features - X.shape[1]
    if n_noise > 0:
        X_noise = rng.standard_normal(size=(X.shape[0], n_noise))
        X = np.hstack([X, X_noise])
    
    if shift > 0:
        X += rng.uniform(-shift, shift, size=X.shape)
    
    if scale > 1:
        X *= rng.uniform(1 - scale, 1 + scale, size=X.shape)
    
    if flip_y > 0:
        n_to_flip = int(n_samples * flip_y)
        indices_to_flip = rng.choice(n_samples, size=n_to_flip, replace=False)
        y[indices_to_flip] = rng.integers(0, n_classes, size=n_to_flip)
    
    if shuffle:
        indices = rng.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        feature_indices = rng.permutation(n_features)
        X = X[:, feature_indices]
    
    return X, y


def log_softmax(x: np.ndarray) -> np.ndarray:
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return x - max_x - np.log(sum_exp)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x))


def format_number(number):
    if number == 0:  # Handle the case for 0 directly
        return "0.0"

    if abs(number) < 1e-3:
        exponent = int(f"{number:.1e}".split("e")[1])
        significant_digits = max(0, abs(exponent)) + 1
        return f"{number:.{significant_digits}e}"
    else:
        return f"{number:.4f}"


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
        raise NotImplementedError(
            f"Display check not implemented for {system}")


def is_display_available_linux():
    if "DISPLAY" in os.environ:
        try:
            output = subprocess.check_output(
                ["xdpyinfo"], stderr=subprocess.STDOUT)
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


class GradientDebugger:
    def __init__(self, clip_threshold: float = 1.0):
        self.stats_history = []
        self.clip_threshold = clip_threshold
        self.running_mean_norm = None
        self.running_std_norm = None
        self.beta = 0.9

    def adaptive_clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        if gradient is None:
            return None

        grad_norm = np.linalg.norm(gradient)
        if self.running_mean_norm is None:
            self.running_mean_norm = grad_norm
            self.running_std_norm = 1.0
        else:
            self.running_mean_norm = self.beta * \
                self.running_mean_norm + (1 - self.beta) * grad_norm
            self.running_std_norm = (self.beta * self.running_std_norm +
                                     (1 - self.beta) * abs(grad_norm - self.running_mean_norm))

        adaptive_threshold = self.running_mean_norm + 2 * self.running_std_norm
        clip_norm = min(self.clip_threshold * adaptive_threshold, 10.0)

        if grad_norm > clip_norm:
            return gradient * (clip_norm / grad_norm)
        return gradient

    def compute_gradient_stats(self, gradient: np.ndarray) -> dict:
        if gradient is None or not isinstance(gradient, np.ndarray):
            return self._empty_stats()

        flat_grad = gradient.flatten()
        grad_norm = float(np.linalg.norm(flat_grad))

        stats = {
            'mean': float(np.mean(flat_grad)),
            'std': float(np.std(flat_grad)),
            'min': float(np.min(flat_grad)),
            'max': float(np.max(flat_grad)),
            'norm': grad_norm,
            'zeros_pct': float(np.sum(np.abs(flat_grad) < 1e-8) / len(flat_grad) * 100),
            'is_valid': bool(not np.any(np.isnan(flat_grad)) and not np.any(np.isinf(flat_grad))),
            'relative_norm': grad_norm / self.running_mean_norm if self.running_mean_norm else 1.0
        }

        hist, _ = np.histogram(flat_grad, bins=10)
        stats['distribution'] = hist.tolist()

        return stats

    def _empty_stats(self) -> dict:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'norm': 0.0, 'zeros_pct': 100.0, 'is_valid': False,
            'relative_norm': 0.0, 'distribution': [0] * 10
        }

    def log_gradient_stats(self, name: str, gradient: np.ndarray, step: int):
        stats = self.compute_gradient_stats(gradient)
        stats['step'] = step
        stats['name'] = name
        self.stats_history.append(stats)

        self._check_gradient_health(name, stats, step)

    def _check_gradient_health(self, name: str, stats: dict, step: int):
        warnings = []

        if not stats['is_valid']:
            warnings.append("Invalid gradients detected")

        if stats['norm'] > 100:
            warnings.append(f"Large gradient norm ({stats['norm']:.2f})")

        if stats['zeros_pct'] > 90:
            warnings.append(
                f"High percentage of zeros ({stats['zeros_pct']:.1f}%)")

        if stats['relative_norm'] > 5.0:
            warnings.append(
                f"Gradient norm {stats['relative_norm']:.1f}x larger than running average")

        if warnings:
            print(f"WARNING in {name} at step {step}:")
            for warning in warnings:
                print(f"- {warning}")

    def get_summary(self, last_n: int = None) -> dict:
        if not self.stats_history:
            return {}

        history = self.stats_history[-last_n:] if last_n else self.stats_history

        grouped = {}
        for entry in history:
            name = entry['name']
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(entry)

        summary = {}
        for name, entries in grouped.items():
            norms = [e['norm'] for e in entries]
            relative_norms = [e['relative_norm'] for e in entries]

            summary[name] = {
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'mean_relative_norm': np.mean(relative_norms),
                'std_relative_norm': np.std(relative_norms),
                'mean_zeros_pct': np.mean([e['zeros_pct'] for e in entries]),
                'invalid_count': sum(1 for e in entries if not e['is_valid']),
                'norm_trend': np.polyfit(range(len(norms)), norms, 1)[0],
                'samples': len(entries)
            }

        return summary
