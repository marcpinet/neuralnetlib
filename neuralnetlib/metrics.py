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


class Metric:
    def __init__(self, name: str):
        if isinstance(name, str):
            self.function = self._get_function_by_name(name)
            self.name = self._get_function_by_name(
                name).__name__.split("_score")[0]
        elif callable(name):
            self.function = name
            self.name = name.__name__.split("_score")[0]

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
        y_pred, y_true = _reshape_inputs(y_pred, y_true)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        return self.function(y_pred, y_true, threshold)


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        return np.mean(y_pred_classes == y_true)
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))


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
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_classes == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred_classes == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    precisions = [
        np.sum((y_pred_classes == cls) & (y_true_classes == cls)) /
        np.sum(y_pred_classes == cls)
        for cls in np.unique(y_true_classes) if np.sum(y_pred_classes == cls) > 0
    ]
    return np.mean(precisions) if precisions else 0.0


def recall_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_classes == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
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
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int).ravel()
        y_true_classes = y_true.ravel()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

    classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
    n_classes = len(classes)
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

    return np.trapz(tpr, fpr)


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

    return np.trapz(precision[sl], recall[sl])


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
        y_pred_classes = (y_pred >= threshold).astype(int).ravel()
        y_true_classes = y_true.ravel()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

    ss_res = np.sum((y_true_classes - y_pred_classes) ** 2)
    ss_tot = np.sum((y_true_classes - np.mean(y_true_classes)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def bleu_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float | None = None, n_gram: int = 4, smooth: bool = False) -> float:
    """Compute BLEU score for machine translation evaluation.

    Args:
        y_pred: Model predictions (batch_size, seq_length, vocab_size) or (batch_size, seq_length)
        y_true: True sequences (batch_size, seq_length)
        threshold: Optional threshold parameter (ignored for BLEU score)
        n_gram: Maximum n-gram length. Defaults to 4.
        smooth: Whether to apply smoothing. Defaults to False.

    Returns:
        float: BLEU score.
    """
    special_tokens = {0, 1, 2, 3}  # PAD, UNK, SOS, EOS
    weights = [0.25] * n_gram

    if y_pred.ndim == 3:
        y_pred = np.argmax(y_pred, axis=-1)

    def filter_special_tokens(seq):
        return [token for token in seq if token not in special_tokens]

    pred_sequences = [filter_special_tokens(
        [int(token) for token in seq]) for seq in y_pred]
    true_sequences = [[filter_special_tokens(
        [int(token) for token in seq])] for seq in y_true]

    def get_ngrams(sequence, n):
        return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

    def smooth_precision(matches, total, n):
        k = 1
        return (matches + k) / (total + k)

    precisions = []
    for n in range(1, int(n_gram) + 1):
        matches = 0
        total = 0
        for pred, refs in zip(pred_sequences, true_sequences):
            if len(pred) < n:
                continue

            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams_list = [get_ngrams(ref, n) for ref in refs]

            pred_count = {}
            for ngram in pred_ngrams:
                pred_count[ngram] = pred_count.get(ngram, 0) + 1

            max_ref_count = {}
            for ref_ngrams in ref_ngrams_list:
                ref_count = {}
                for ngram in ref_ngrams:
                    ref_count[ngram] = ref_count.get(ngram, 0) + 1
                for ngram, count in ref_count.items():
                    max_ref_count[ngram] = max(
                        max_ref_count.get(ngram, 0), count)

            for ngram, count in pred_count.items():
                matches += min(count, max_ref_count.get(ngram, 0))
            total += len(pred_ngrams)

        if smooth:
            precisions.append(smooth_precision(matches, total, n))
        else:
            precisions.append(matches / total if total > 0 else 0)

    pred_length = sum(len(pred) for pred in pred_sequences)
    ref_length = sum(min(len(ref) for ref in refs) for refs in true_sequences)

    brevity_penalty = np.exp(
        min(1 - ref_length/pred_length, 0)) if pred_length > 0 else 0

    if all(p == 0 for p in precisions):
        return 0.0

    weighted_scores = [w * np.log(p if p > 0 else 1e-10)
                       for w, p in zip(weights, precisions)]
    bleu = brevity_penalty * np.exp(sum(weighted_scores))

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

    recall_total = 0
    precision_total = 0
    for pred, refs in zip(y_pred, y_true):
        pred_ngrams = get_ngrams(pred, n)
        ref_ngrams_list = [get_ngrams(ref, n) for ref in refs]

        pred_count = len(pred_ngrams)
        max_matches = 0

        for ref_ngrams in ref_ngrams_list:
            ref_count = len(ref_ngrams)
            matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
            max_matches = max(max_matches, matches)

        recall_total += max_matches / ref_count if ref_count > 0 else 0
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


def mmd_score(y_pred: np.ndarray, y_true: np.ndarray, sigma: float = None) -> float:
    y_pred = y_pred.reshape(len(y_pred), -1)
    y_true = y_true.reshape(len(y_true), -1)

    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
        dist_matrix = x_norm + y_norm - 2 * np.dot(x, y.T)
        return np.exp(-dist_matrix / (2 * sigma ** 2))

    if sigma is None:
        combined = np.vstack((y_pred, y_true))
        pairwise_dists = np.linalg.norm(combined[:, None] - combined, axis=2)
        sigma = np.median(pairwise_dists)

    n = len(y_pred)
    m = len(y_true)

    k_xx = gaussian_kernel(y_pred, y_pred, sigma)
    k_yy = gaussian_kernel(y_true, y_true, sigma)
    k_xy = gaussian_kernel(y_pred, y_true, sigma)

    xx_term = (np.sum(k_xx) - np.sum(np.diag(k_xx))) / \
        (n * (n - 1)) if n > 1 else 0
    yy_term = (np.sum(k_yy) - np.sum(np.diag(k_yy))) / \
        (m * (m - 1)) if m > 1 else 0
    xy_term = np.sum(k_xy) / (n * m)

    return float(xx_term + yy_term - 2 * xy_term)


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

    x_beta = df / (df + t_stat**2)
    p_value = 2 * regularized_incomplete_beta(df / 2, 0.5, x_beta)

    return np.array(r), np.array(p_value)


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ln_term = a * np.log(x) + b * np.log(1 - x)
    sum_term = 1.0
    term = 1.0

    for k in range(1, 100):
        term *= (a + k - 1) * x / (k * (b + k - 1))
        sum_term += term
        if term < 1e-15:
            break

    result = np.exp(ln_term) * sum_term / a
    return result


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
    
    kurt = (n * m4) / (m2**2)
    
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

    skewness = (n * m3) / (m2**1.5)
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
    predictions = (y_pred >= threshold).astype(int)
    
    intersection = np.sum(predictions & y_true, axis=1)
    union = np.sum(predictions | y_true, axis=1)
    
    return np.mean(intersection / (union + 1e-15))


def precision_at_k(y_pred: np.ndarray, y_true: np.ndarray, k: int) -> float:
    batch_size = y_true.shape[0]
    topk_pred = np.zeros_like(y_pred)
    
    for i in range(batch_size):
        top_k_indices = np.argsort(y_pred[i])[-k:]
        topk_pred[i, top_k_indices] = 1
        
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
        
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    
    for i in range(len(y_true)):
        contingency[np.nonzero(classes == y_true[i])[0][0],
                   np.nonzero(clusters == y_pred[i])[0][0]] += 1
                   
    contingency = contingency.astype(np.float64)
    
    a = np.sum(contingency, axis=1)
    b = np.sum(contingency, axis=0)
    n = np.sum(contingency)
    
    eps = np.finfo(float).eps
    h_true = -np.sum((a[a > 0] / n) * np.log(a[a > 0] / n))
    h_pred = -np.sum((b[b > 0] / n) * np.log(b[b > 0] / n))
    
    MI = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if contingency[i, j] > 0:
                MI += (contingency[i, j] / n) * \
                      np.log((contingency[i, j] * n) / (a[i] * b[j]))
                      
    if h_true == 0 or h_pred == 0:
        return 0.0
        
    denominator = (h_true + h_pred) / 2
    if denominator < eps:
        return 0.0
        
    nmi = MI / denominator
    
    return float(np.clip(nmi, 0.0, 1.0))
