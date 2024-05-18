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
        self.best_metric = None
        self.patience_counter = 0
        self.epoch = 0
        self.stop_training = False

    def on_epoch_end(self, model, loss, metrics=None):
        self.epoch += 1
        if self.epoch < self.start_from_epoch:
            return False

        if self.best_metric is None:
            if self.monitor is None:
                self.best_metric = loss
                self.mode = 'min'
            else:
                if metrics is None:
                    raise ValueError("Metric to monitor is provided, but no metrics are available.")
                metric_value = metrics[self.monitor[0].__name__]
                self.best_metric = metric_value
                if self.mode == 'auto':
                    if np.isnan(metric_value):
                        self.mode = 'min'
                    else:
                        self.mode = 'max'

        improved = False
        if self.monitor is None:
            current_metric = loss
            if (self.mode == 'min' and current_metric < self.best_metric - self.min_delta) or \
                    (self.mode == 'max' and current_metric > self.best_metric + self.min_delta):
                self.best_metric = current_metric
                improved = True
        else:
            current_metric = metrics[self.monitor[0].__name__]
            if (self.mode == 'max' and current_metric > self.best_metric + self.min_delta) or \
                    (self.mode == 'min' and current_metric < self.best_metric - self.min_delta):
                self.best_metric = current_metric
                improved = True

        if improved:
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = [layer.weights for layer in model.layers if hasattr(layer, 'weights')]
        else:
            self.patience_counter += 1
            if self.baseline is not None:
                if self.mode == 'max' and self.best_metric < self.baseline:
                    self.patience_counter = self.patience + 1
                elif self.mode == 'min' and self.best_metric > self.baseline:
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
