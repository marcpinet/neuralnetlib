import numpy as np

from neuralnetlib.preprocessing import apply_threshold

import numpy as np
from neuralnetlib.preprocessing import apply_threshold

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
        else:
            raise ValueError(f"Metric {name} is not supported.")
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            
        return self.function(y_pred, y_true, threshold)

def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        return np.mean(y_pred_classes == y_true)
    
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def precision_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_classes == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred_classes == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    classes = np.unique(y_true_classes)
    
    precisions = []
    for cls in classes:
        true_positives = np.sum((y_pred_classes == cls) & (y_true_classes == cls))
        predicted_positives = np.sum(y_pred_classes == cls)
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
            
    return np.mean(precisions) if precisions else 0.0

def recall_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        
    if y_pred.shape[1] == 1:
        y_pred_classes = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_classes == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    classes = np.unique(y_true_classes)
    
    recalls = []
    for cls in classes:
        true_positives = np.sum((y_pred_classes == cls) & (y_true_classes == cls))
        actual_positives = np.sum(y_true_classes == cls)
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
            
    return np.mean(recalls) if recalls else 0.0

def f1_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    precision = precision_score(y_pred, y_true, threshold)
    recall = recall_score(y_pred, y_true, threshold)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        
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