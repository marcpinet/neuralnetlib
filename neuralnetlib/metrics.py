import numpy as np


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
            self.name = self._get_function_by_name(name).__name__.split("_score")[0]
        elif callable(name):
            self.function = name
            self.name = name.__name__.split("_score")[0]

    def _get_function_by_name(self, name: str):
        if name in ['accuracy', 'accuracy_score', 'accuracy-score', 'acc']:
            return accuracy_score
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
        np.sum((y_pred_classes == cls) & (y_true_classes == cls)) / np.sum(y_pred_classes == cls)
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
        np.sum((y_pred_classes == cls) & (y_true_classes == cls)) / np.sum(y_true_classes == cls)
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int).ravel()
        y_true_classes = y_true.ravel()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

    return np.mean((y_pred_classes - y_true_classes) ** 2)


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int).ravel()
        y_true_classes = y_true.ravel()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
    
    return np.mean(np.abs(y_pred_classes - y_true_classes))


def mean_absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred, y_true = _reshape_inputs(y_pred, y_true)
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int).ravel()
        y_true_classes = y_true.ravel()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
    
    mask = y_true_classes != 0
    return np.mean(np.abs((y_true_classes[mask] - y_pred_classes[mask]) / y_true_classes[mask])) * 100


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