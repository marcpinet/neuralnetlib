import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True,
                 start_from_epoch: int = 0, monitor: list = None, mode: str = 'auto', baseline: float = None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.monitor = monitor
        self.mode = mode
        self.baseline = baseline
        self.best_weights = None
        self.best_metrics = None
        self.patience_counter = 0
        self.epoch = 0
        self.stop_training = False

    def on_epoch_end(self, model, metrics):
        self.epoch += 1
        if self.epoch < self.start_from_epoch:
            return False

        if self.best_metrics is None:
            if self.monitor is None:
                self.best_metrics = metrics
                if np.any(np.isnan(metrics)):
                    self.mode = 'min'
                else:
                    self.mode = 'auto'
            else:
                metric_values = [metric(model.predictions, model.y_true) for metric in self.monitor]
                self.best_metrics = [np.inf if m > 0 else -np.inf for m in metric_values]
                self.mode = 'max'

        improved = False
        if self.monitor is None:
            current_metric = metrics[-1]
            best_metric = self.best_metrics[-1]
            if (self.mode == 'min' and current_metric < best_metric - self.min_delta) or \
               (self.mode == 'max' and current_metric > best_metric + self.min_delta) or \
               (self.mode == 'auto' and current_metric < best_metric - self.min_delta):
                self.best_metrics[-1] = current_metric
                improved = True
        else:
            for i, metric in enumerate(metrics):
                best_metric = self.best_metrics[i]
                if (self.mode == 'max' and metric > best_metric + self.min_delta) or \
                   (self.mode == 'min' and metric < best_metric - self.min_delta):
                    self.best_metrics[i] = metric
                    improved = True

        if improved:
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = [layer.weights for layer in model.layers if hasattr(layer, 'weights')]
        else:
            self.patience_counter += 1

        if self.baseline is not None:
            if self.mode == 'max' and max(self.best_metrics) < self.baseline:
                self.patience_counter = self.patience + 1
            elif self.mode == 'min' and min(self.best_metrics) > self.baseline:
                self.patience_counter = self.patience + 1

        if self.patience_counter >= self.patience:
            self.stop_training = True
            print(f"\nEarly stopping after {self.epoch} epochs.", end='')
            if self.restore_best_weights and self.best_weights is not None:
                for layer, best_weights in zip([layer for layer in model.layers if hasattr(layer, 'weights')],
                                               self.best_weights):
                    layer.weights = best_weights
            return True

        return False