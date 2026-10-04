import inspect
import math

import numpy as np

from collections import namedtuple


def _reshape_inputs(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    return y_pred, y_true


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapz was removed from recent NumPy versions (replaced by np.trapezoid)
    integrate = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(integrate(y, x))


class Metric:
    def __init__(self, name: str):
        if isinstance(name, Metric):
            self.function = name.function
            self.name = name.name
        elif isinstance(name, str):
            self.function = self._get_function_by_name(name)
            self.name = self.function.__name__.split("_score")[0]
        elif callable(name):
            self.function = name
            self.name = getattr(name, '__name__', type(name).__name__).split("_score")[0]
        else:
            raise ValueError(f"Metric {name} is not supported.")

    def _get_function_by_name(self, name: str):
        if name in ['accuracy', 'accuracy_score', 'accuracy-score', 'acc']:
            return accuracy_score
        elif name in ['sparse_categorical_accuracy', 'sparse-categorical-accuracy', 'sparse_acc']:
            return sparse_categorical_accuracy_score
        elif name in ['f1', 'f1_score', 'f1-score']:
            return f1_score
        elif name in ['recall', 'recall_score', 'recall-score', 'sensitivity', 'rec']:
            return recall_score
        elif name in ['precision', 'precision_score', 'precision-score', 'positive-predictive-value']:
            return precision_score
        elif name in ['roc-auc', 'roc_auc', 'roc-auc-score']:
            return roc_auc_score
        elif name in ['pr-auc', 'pr_auc', 'pr-auc-score']:
            return pr_auc_score
        elif name in ['mean-squared-error', 'mse']:
            return mean_squared_error
        elif name in ['mean-absolute-error', 'mae']:
            return mean_absolute_error
        elif name in ['mean-absolute-percentage-error', 'mape']:
            return mean_absolute_percentage_error
        elif name in ['r2', 'r2_score']:
            return r2_score
        elif name in ['bleu', 'bleu_score']:
            return bleu_score
        elif name in ['rouge-n', 'rouge_n', 'rouge-n-score']:
            return rouge_n_score
        elif name in ['rouge-l', 'rouge_l', 'rouge-l-score']:
            return rouge_l_score
        elif name in ['mmd', 'mmd_score', 'maximum-mean-discrepancy']:
            return mmd_score
        elif name in ['hamming-loss', 'hamming_loss', 'hamming']:
            return hamming_loss
        elif name in ['exact-match-ratio', 'exact_match_ratio', 'exact-match']:
            return exact_match_ratio
        elif name in ['jaccard-similarity', 'jaccard_similarity', 'jaccard']:
            return jaccard_similarity
        elif name in ['subset-accuracy', 'subset_accuracy', 'subset']:
            return subset_accuracy
        elif name in ['precision-at-k', 'precision_at_k', 'precision-k']:
            return precision_at_k
        elif name in ['f1-score-per-label', 'f1_score_per_label', 'f1-per-label']:
            return f1_score_per_label
        else:
            raise ValueError(f"Metric {name} is not supported.")

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        try:
            y_pred, y_true = _reshape_inputs(y_pred, y_true)
        except ValueError:
            # ragged inputs (e.g. lists of tokens for the ROUGE scores) are given as is
            pass

        # the threshold is only given to the metrics using one (it would be taken as another parameter otherwise,
        # e.g. the n-gram size of ROUGE-N or the sigma of the MMD)
        try:
            parameters = inspect.signature(self.function).parameters
        except (TypeError, ValueError):
            parameters = {}
        if 'threshold' in parameters:
            return self.function(y_pred, y_true, threshold=threshold)
        return self.function(y_pred, y_true)


def _class_predictions(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true classes, whatever the format of the labels: binary probabilities, one-hot encoded labels,
    or class indices (sparse labels). Sequences of predictions (batch_size, timesteps, n_classes) are supported."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_pred.shape[-1] == 1:
        return (y_pred >= threshold).astype(int).reshape(-1), y_true.reshape(-1)

    pred_classes = np.argmax(y_pred, axis=-1)
    if y_true.shape == y_pred.shape:
        true_classes = np.argmax(y_true, axis=-1)
    else:
        # class indices, possibly with a trailing axis of size 1
        true_classes = y_true.reshape(pred_classes.shape)
    return pred_classes.reshape(-1), true_classes.reshape(-1)


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_classes, y_true_classes = _class_predictions(y_pred, y_true, threshold)
    return float(np.mean(y_pred_classes == y_true_classes))


def sparse_categorical_accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.ndim > 1:
        if y_true.shape[1] == 1:
            y_true = y_true.ravel()
        else:
            raise ValueError(
                "y_true should be a 1D array of shape (n_samples,) containing integer class indices")

    predicted_classes = np.argmax(y_pred, axis=1)

    return np.mean(predicted_classes == y_true)


def precision_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    y_pred_classes, y_true_classes = _class_predictions(y_pred, y_true, threshold)
    if y_pred.shape[-1] == 1:
        true_positives = np.sum((y_pred_classes == 1) & (y_true_classes == 1))
        predicted_positives = np.sum(y_pred_classes == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

    precisions = [
        np.sum((y_pred_classes == cls) & (y_true_classes == cls)) /
        np.sum(y_pred_classes == cls)
        for cls in np.unique(y_true_classes) if np.sum(y_pred_classes == cls) > 0
    ]
    return np.mean(precisions) if precisions else 0.0


def recall_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    y_pred_classes, y_true_classes = _class_predictions(y_pred, y_true, threshold)
    if y_pred.shape[-1] == 1:
        true_positives = np.sum((y_pred_classes == 1) & (y_true_classes == 1))
        actual_positives = np.sum(y_true_classes == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

    recalls = [
        np.sum((y_pred_classes == cls) & (y_true_classes == cls)) /
        np.sum(y_true_classes == cls)
        for cls in np.unique(y_true_classes) if np.sum(y_true_classes == cls) > 0
    ]
    return np.mean(recalls) if recalls else 0.0


def f1_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    precision = precision_score(y_pred, y_true, threshold)
    recall = recall_score(y_pred, y_true, threshold)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    y_pred_classes, y_true_classes = _class_predictions(y_pred, y_true, threshold)
    y_pred_classes = y_pred_classes.astype(int)
    y_true_classes = y_true_classes.astype(int)

    # the classes are used as indices: the matrix must cover all of them, even the ones never seen in this data
    n_classes = int(max(np.max(y_true_classes), np.max(y_pred_classes), y_pred.shape[-1] - 1, 1)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true_classes)):
        cm[y_true_classes[i], y_pred_classes[i]] += 1

    return cm


def classification_report(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> str:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    cm = confusion_matrix(y_pred, y_true, threshold)
    n_classes = cm.shape[0]

    metrics = {
        'precision': np.zeros(n_classes),
        'recall': np.zeros(n_classes),
        'f1': np.zeros(n_classes),
        'support': np.zeros(n_classes)
    }

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0

        metrics['precision'][i] = precision
        metrics['recall'][i] = recall
        metrics['f1'][i] = f1
        metrics['support'][i] = support

    # Calculate macro averages
    macro_precision = np.mean(metrics['precision'])
    macro_recall = np.mean(metrics['recall'])
    macro_f1 = np.mean(metrics['f1'])
    total_support = np.sum(metrics['support'])

    # Format report
    report = "Classification Report\n"
    report += "=" * 70 + "\n"
    report += f"{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}\n"
    report += "-" * 70 + "\n"

    # Add per-class metrics
    for i in range(n_classes):
        report += f"{i:>8d} {metrics['precision'][i]:>10.2f} {metrics['recall'][i]:>10.2f} "
        report += f"{metrics['f1'][i]:>10.2f} {metrics['support'][i]:>10.0f}\n"

    # Add macro averages
    report += "\n"
    report += f"{'macro avg':>8} {macro_precision:>10.2f} {macro_recall:>10.2f} "
    report += f"{macro_f1:>10.2f} {total_support:>10.0f}\n"

    return report


def roc_auc_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)

    if y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
    else:
        raise ValueError("Multiclass ROC AUC not implemented yet.")

    if len(np.unique(y_true)) != 2:
        return 0.0

    desc_score_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.nonzero(np.diff(y_pred[desc_score_indices]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    return _trapezoid(tpr, fpr)


def pr_auc_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)

    if y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
    else:
        raise ValueError("Multiclass PR AUC not implemented yet.")

    if len(np.unique(y_true)) != 2:
        return 0.0

    desc_score_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.nonzero(np.diff(y_pred[desc_score_indices]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    precision = np.r_[1, precision]
    recall = np.r_[0, recall]

    last_ind = precision.size
    sl = slice(0, last_ind)

    return _trapezoid(precision[sl], recall[sl])


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    return np.mean((y_pred - y_true) ** 2)


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    return np.mean(np.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    mask = np.abs(y_true) > 1e-10
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)

    if y_pred.shape[1] == 1:
        y_pred_ = y_pred.ravel()
        y_true_ = y_true.ravel()
        ss_res = np.sum((y_true_ - y_pred_) ** 2)
        ss_tot = np.sum((y_true_ - np.mean(y_true_)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    r2s = []
    for j in range(y_pred.shape[1]):
        yp = y_pred[:, j]
        yt = y_true[:, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2s.append(1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0)
    return float(np.mean(r2s)) if r2s else 0.0


def bleu_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float | None = None, 
               n_gram: int = 4, smooth: bool = False) -> float:
    """Compute BLEU score for machine translation evaluation."""

    special_tokens = {0, 1, 2, 3}  # PAD, UNK, SOS, EOS
    weights = [1.0/n_gram] * n_gram

    if y_pred.ndim == 3:
        y_pred = np.argmax(y_pred, axis=-1)
    
    def filter_special_tokens(seq):
        return [token for token in seq if token not in special_tokens]
    
    pred_sequences = [filter_special_tokens([int(token) for token in seq]) for seq in y_pred]
    true_sequences = [filter_special_tokens([int(token) for token in seq]) for seq in y_true]
    
    def get_ngrams(sequence, n):
        if len(sequence) < n:
            return []
        return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
    
    def smooth_precision(matches, total, n):
        return (matches + 1) / (total + 1)

    precisions = []
    for n in range(1, n_gram + 1):
        matches = 0
        total = 0
        
        for pred, ref in zip(pred_sequences, true_sequences):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            if not pred_ngrams:
                continue
                
            pred_count = {}
            for ngram in pred_ngrams:
                pred_count[ngram] = pred_count.get(ngram, 0) + 1
                
            ref_count = {}
            for ngram in ref_ngrams:
                ref_count[ngram] = ref_count.get(ngram, 0) + 1
            
            for ngram, count in pred_count.items():
                matches += min(count, ref_count.get(ngram, 0))
            
            total += len(pred_ngrams)
        
        if smooth and total > 0:
            precisions.append(smooth_precision(matches, total, n))
        else:
            precisions.append(matches / total if total > 0 else 0.0)
    
    pred_length = sum(len(seq) for seq in pred_sequences)
    ref_length = sum(len(seq) for seq in true_sequences)

    if pred_length == 0:
        return 0.0
        
    brevity_penalty = min(1.0, np.exp(1 - ref_length/pred_length))
    
    if all(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [w * np.log(max(p, 1e-10)) for w, p in zip(weights, precisions)]
    bleu = brevity_penalty * np.exp(sum(log_precisions))
    
    return float(bleu)


def rouge_n_score(y_pred: list[list[str]], y_true: list[list[list[str]]], n: int = 2) -> float:
    """Compute ROUGE-N score for text summarization evaluation.

    Args:
        y_pred (list[list[str]]): List of list containing predicted tokens.
        y_true (list[list[list[str]]]): List of list containing reference tokens.
        n (int, optional): Maximum n-gram length. Defaults to 2.

    Returns:
        float: ROUGE-N score.
    """
    def get_ngrams(sequence, n):
        return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

    def count_matches(pred_ngrams, ref_ngrams):
        # clipped counts: an n-gram of the prediction cannot match more times than it appears in the reference
        ref_counts = {}
        for ngram in ref_ngrams:
            ref_counts[ngram] = ref_counts.get(ngram, 0) + 1
        matches = 0
        for ngram in pred_ngrams:
            if ref_counts.get(ngram, 0) > 0:
                ref_counts[ngram] -= 1
                matches += 1
        return matches

    recall_total = 0
    precision_total = 0
    for pred, refs in zip(y_pred, y_true):
        pred_ngrams = get_ngrams(pred, n)
        ref_ngrams_list = [get_ngrams(ref, n) for ref in refs]

        pred_count = len(pred_ngrams)
        max_matches = 0
        best_ref_count = 0

        for ref_ngrams in ref_ngrams_list:
            matches = count_matches(pred_ngrams, ref_ngrams)
            if matches > max_matches or (matches == max_matches and best_ref_count == 0):
                max_matches = matches
                best_ref_count = len(ref_ngrams)

        # the recall is computed with the reference giving the best match
        recall_total += max_matches / best_ref_count if best_ref_count > 0 else 0
        precision_total += max_matches / pred_count if pred_count > 0 else 0

    recall_avg = recall_total / len(y_pred)
    precision_avg = precision_total / len(y_pred)
    rouge_n = 2 * (recall_avg * precision_avg) / (recall_avg +
                                                  precision_avg) if (recall_avg + precision_avg) > 0 else 0
    return rouge_n


def rouge_l_score(y_pred: list[list[str]], y_true: list[list[list[str]]]) -> float:
    """Compute ROUGE-L score for text summarization evaluation.

    Args:
        y_pred (list[list[str]]): List of list containing predicted tokens.
        y_true (list[list[list[str]]]): List of list containing reference tokens.

    Returns:
        float: ROUGE-L score.
    """
    def lcs_length(x, y):
        dp = np.zeros((len(x) + 1, len(y) + 1), dtype=int)
        for i in range(1, len(x) + 1):
            for j in range(1, len(y) + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    recall_total = 0
    precision_total = 0
    for pred, refs in zip(y_pred, y_true):
        lcs_max = 0
        ref_lengths = []
        for ref in refs:
            lcs_max = max(lcs_max, lcs_length(pred, ref))
            ref_lengths.append(len(ref))

        recall_total += lcs_max / \
            max(ref_lengths) if max(ref_lengths) > 0 else 0
        precision_total += lcs_max / len(pred) if len(pred) > 0 else 0

    recall_avg = recall_total / len(y_pred)
    precision_avg = precision_total / len(y_pred)
    rouge_l = 2 * (recall_avg * precision_avg) / (recall_avg +
                                                  precision_avg) if (recall_avg + precision_avg) > 0 else 0
    return rouge_l


def mmd_score(y_pred: np.ndarray, y_true: np.ndarray, sigma: float = None, random_state: float = None) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = y_pred.reshape(len(y_pred), -1)
    y_true = y_true.reshape(len(y_true), -1)

    # both sets are scaled to [-1, 1] with the same range: normalizing them independently would hide their differences
    x_min = min(y_pred.min(), y_true.min())
    x_max = max(y_pred.max(), y_true.max())
    y_pred = 2 * (y_pred - x_min) / (x_max - x_min + 1e-8) - 1
    y_true = 2 * (y_true - x_min) / (x_max - x_min + 1e-8) - 1

    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        x_squared = np.sum(x**2, axis=1, keepdims=True)
        y_squared = np.sum(y**2, axis=1, keepdims=True).T
        xy = np.dot(x, y.T)
        dist_matrix = x_squared + y_squared - 2 * xy
        dist_matrix = np.clip(dist_matrix, 0, None)
        return np.exp(-dist_matrix / (2 * sigma**2))

    if sigma is None:
        # median heuristic on the pooled samples
        pooled = np.concatenate([y_pred, y_true])
        n_samples = min(1000, len(pooled))
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(pooled), n_samples, replace=False)
        subset = pooled[indices]
        dists = np.linalg.norm(subset[:, np.newaxis] - subset, axis=2)
        positive_dists = dists[dists > 0]
        sigma = np.median(positive_dists) if positive_dists.size > 0 else 1.0
        if sigma < 1e-10:
            sigma = 1.0
    
    n = len(y_pred)
    m = len(y_true)
    
    k_xx = gaussian_kernel(y_pred, y_pred, sigma)
    k_yy = gaussian_kernel(y_true, y_true, sigma)
    k_xy = gaussian_kernel(y_pred, y_true, sigma)
    
    eps = 1e-8
    xx_term = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1) + eps)
    yy_term = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1) + eps)
    xy_term = np.mean(k_xy)
    
    mmd = xx_term + yy_term - 2 * xy_term
    
    return float(np.clip(mmd, 0, None))


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1D.")
    if x.size != y.size:
        raise ValueError("Arrays must have the same size.")
    if x.size < 2:
        raise ValueError("Arrays must have at least 2 elements.")
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator == 0:
        return np.array(0.0), np.array(1.0)

    r = numerator / denominator

    if np.isclose(r, 1.0, rtol=1e-09, atol=1e-09) or r == -1.0:
        return np.array(r), np.array(0.0)

    n = x.size
    df = n - 2
    t_stat = r * np.sqrt(df / (1 - r**2))

    # two-sided p-value of the Student's t-test: P(|T| >= |t|) = I_x(df / 2, 1 / 2) with x = df / (df + t^2)
    x_beta = df / (df + t_stat**2)
    p_value = regularized_incomplete_beta(df / 2, 0.5, x_beta)

    return np.array(r), np.array(p_value)


def _beta_continued_fraction(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-16) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz's method, Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
                 a * math.log(x) + b * math.log(1 - x))
    front = math.exp(log_front)

    # the continued fraction converges quickly for x < (a + 1) / (a + b + 2), the symmetry relation is used otherwise
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1 - x) / b


def kurtosis(x: np.ndarray, fisher: bool = True) -> float:
    if x.ndim != 1:
        raise ValueError("Input array must be 1D.")
    if x.size < 2:
        raise ValueError("Array must have at least 2 elements.")

    n = x.size
    mean = np.mean(x)
    deviations = x - mean
    m2 = np.mean(deviations**2)
    m4 = np.mean(deviations**4)

    if m2 <= 1e-15:
        return np.nan

    # m2 and m4 are already averaged over the n samples
    kurt = m4 / (m2**2)

    if fisher:
        kurt -= 3

    return kurt


def skew(x: np.ndarray) -> float:
    if x.ndim != 1:
        raise ValueError("Input array must be 1D.")
    if x.size < 2:
        raise ValueError("Array must have at least 2 elements.")

    n = x.size
    mean = np.mean(x)
    deviations = x - mean
    m2 = np.mean(deviations**2)
    m3 = np.mean(deviations**3)

    if m2 <= 1e-15:
        return np.nan

    # m2 and m3 are already averaged over the n samples
    skewness = m3 / (m2**1.5)
    return skewness


def hamming_loss(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    predictions = (y_pred >= threshold).astype(int)
    return np.mean(predictions != y_true)


def exact_match_ratio(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    predictions = (y_pred >= threshold).astype(int)
    return np.mean(np.all(predictions == y_true, axis=1))


def f1_score_per_label(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predictions = (y_pred >= threshold).astype(int)
    
    true_positives = np.sum((predictions == 1) & (y_true == 1), axis=0)
    false_positives = np.sum((predictions == 1) & (y_true == 0), axis=0)
    false_negatives = np.sum((predictions == 0) & (y_true == 1), axis=0)
    
    precision = true_positives / (true_positives + false_positives + 1e-15)
    recall = true_positives / (true_positives + false_negatives + 1e-15)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return f1


def subset_accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    predictions = (y_pred >= threshold).astype(int)
    return np.mean(np.all(predictions == y_true, axis=1))


def jaccard_similarity(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    predictions = np.asarray(y_pred) >= threshold
    y_true = np.asarray(y_true).astype(bool)

    intersection = np.sum(predictions & y_true, axis=1)
    union = np.sum(predictions | y_true, axis=1)

    return np.mean(intersection / (union + 1e-15))


def precision_at_k(y_pred: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).astype(bool)
    k = int(min(k, y_pred.shape[1]))

    # the k labels with the highest scores of each sample are predicted
    topk_pred = np.zeros(y_pred.shape, dtype=bool)
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    np.put_along_axis(topk_pred, top_k_indices, True, axis=1)

    true_positives = np.sum(topk_pred & y_true, axis=1)
    return np.mean(true_positives / k)


def adjusted_rand_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the Adjusted Rand Index between two clusterings.
    
    Args:
        y_pred: array-like of shape (n_samples,), predicted cluster labels
        y_true: array-like of shape (n_samples,), ground truth cluster labels
    
    Returns:
        float: Adjusted Rand Index score (-1 to 1)
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if len(y_pred) != len(y_true):
        raise ValueError("Input arrays must have the same length")
    
    n_samples = len(y_true)
    
    if np.array_equal(y_pred, y_true):
        return 1.0
    
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    
    for i, label in enumerate(classes):
        for j, cluster in enumerate(clusters):
            contingency[i, j] = np.sum((y_true == label) & (y_pred == cluster))
    
    nij = np.sum(contingency * (contingency - 1)) // 2
    
    a = np.sum(contingency, axis=1)
    b = np.sum(contingency, axis=0)
    
    rsum = np.sum(a * (a - 1)) // 2
    csum = np.sum(b * (b - 1)) // 2
    expected = (rsum * csum) / (n_samples * (n_samples - 1) / 2)
    
    max_index = (rsum + csum) / 2
    
    if max_index == expected:
        return 0.0
    
    return (nij - expected) / (max_index - expected)


def adjusted_mutual_info_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the Adjusted Mutual Information between two clusterings.

    Args:
        y_pred (np.ndarray): Predicted cluster labels
        y_true (np.ndarray): Ground truth cluster labels

    Raises:
        ValueError: Input arrays must be 1-dimensional
        ValueError: Input arrays must have the same length

    Returns:
        float: Adjusted Mutual Information score
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if len(y_pred) != len(y_true):
        raise ValueError("Input arrays must have the same length")
        
    if np.array_equal(y_true, y_pred):
        return 1.0

    classes, class_indices = np.unique(y_true, return_inverse=True)
    clusters, cluster_indices = np.unique(y_pred, return_inverse=True)

    # a single cluster in both labelings (or one cluster per sample in both) is a perfect match
    if len(classes) == len(clusters) == 1 or len(classes) == len(clusters) == len(y_true):
        return 1.0

    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(contingency, (class_indices, cluster_indices), 1)

    a = np.sum(contingency, axis=1)
    b = np.sum(contingency, axis=0)
    n = int(np.sum(contingency))

    eps = np.finfo(float).eps
    h_true = -np.sum((a / n) * np.log(a / n))
    h_pred = -np.sum((b / n) * np.log(b / n))

    nonzero = contingency > 0
    outer = np.outer(a, b).astype(np.float64)
    MI = np.sum((contingency[nonzero] / n) * np.log(contingency[nonzero] * n / outer[nonzero]))

    # expected mutual information under the hypergeometric model of randomness (Vinh et al., 2009)
    log_factorials = np.zeros(n + 1)
    log_factorials[1:] = np.cumsum(np.log(np.arange(1, n + 1)))
    expected_MI = 0.0
    for a_i in a:
        for b_j in b:
            n_ij = np.arange(max(1, a_i + b_j - n), min(a_i, b_j) + 1)
            if n_ij.size == 0:
                continue
            log_probability = (log_factorials[a_i] + log_factorials[b_j] + log_factorials[n - a_i] +
                               log_factorials[n - b_j] - log_factorials[n] - log_factorials[n_ij] -
                               log_factorials[a_i - n_ij] - log_factorials[b_j - n_ij] -
                               log_factorials[n - a_i - b_j + n_ij])
            expected_MI += np.sum((n_ij / n) * np.log(n * n_ij / (a_i * b_j)) * np.exp(log_probability))

    normalizer = (h_true + h_pred) / 2
    denominator = normalizer - expected_MI
    if denominator < 0:
        denominator = min(denominator, -eps)
    else:
        denominator = max(denominator, eps)

    return float((MI - expected_MI) / denominator)
