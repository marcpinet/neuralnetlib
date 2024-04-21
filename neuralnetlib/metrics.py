import numpy as np

from neuralnetlib.preprocessing import apply_threshold


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:  # Binary classification
        y_pred_classes = apply_threshold(y_pred, threshold).ravel()
    else:  # Multiclass classification-regression
        y_pred_classes = np.argmax(y_pred, axis=1)

    if y_true.ndim == 1 or y_true.shape[1] == 1:  # If y_true is not one-hot encoded
        y_true_classes = y_true.ravel()
    else:
        y_true_classes = np.argmax(y_true, axis=1)

    return np.mean(y_pred_classes == y_true_classes)


def f1_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    precision = precision_score(y_pred, y_true, threshold)
    recall = recall_score(y_pred, y_true, threshold)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def recall_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_labels = apply_threshold(y_pred, threshold) if y_pred.shape[1] == 1 else np.argmax(y_pred, axis=1)
    y_true_labels = y_true if y_true.ndim == 1 or y_true.shape[1] == 1 else np.argmax(y_true, axis=1)
    classes = np.unique(y_true_labels)
    recall_scores = []

    for cls in classes:
        tp = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        fn = np.sum((y_pred_labels != cls) & (y_true_labels == cls))

        recall = tp / (tp + fn) if tp + fn != 0 else 0
        recall_scores.append(recall)

    return np.mean(recall_scores)


def precision_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_labels = apply_threshold(y_pred, threshold) if y_pred.shape[1] == 1 else np.argmax(y_pred, axis=1)
    y_true_labels = y_true if y_true.ndim == 1 or y_true.shape[1] == 1 else np.argmax(y_true, axis=1)
    classes = np.unique(y_true_labels)
    precision_scores = []

    for cls in classes:
        tp = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        fp = np.sum((y_pred_labels == cls) & (y_true_labels != cls))

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        precision_scores.append(precision)

    return np.mean(precision_scores)


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:  # Binary classification
        y_pred_classes = apply_threshold(y_pred, threshold).ravel()
    else:  # Multiclass classification-regression
        y_pred_classes = np.argmax(y_pred, axis=1)

    if y_true.ndim == 1 or y_true.shape[1] == 1:  # If y_true is not one-hot encoded
        y_true_classes = y_true.ravel()
    else:
        y_true_classes = np.argmax(y_true, axis=1)

    classes = np.unique(np.concatenate((y_true_classes, y_pred_classes)))
    num_classes = len(classes)

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true_classes)):
        true_class = y_true_classes[i]
        pred_class = y_pred_classes[i]
        cm[true_class, pred_class] += 1

    return cm
