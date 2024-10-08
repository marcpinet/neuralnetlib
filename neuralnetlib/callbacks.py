from neuralnetlib.metrics import Metric


class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True,
                 start_from_epoch: int = 0, monitor: str = 'loss', mode: str = 'auto', baseline: float = None):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.monitor = Metric(monitor) if monitor != 'loss' else 'loss'
        self.mode = mode
        self.baseline = baseline
        self.best_weights = None
        self.best_metric = None
        self.patience_counter = 0
        self.stop_training = False

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.best_metric = None
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        model = logs.get('model')
        if epoch < self.start_from_epoch or model is None:
            return False

        current_metric = self._get_monitor_value(logs)

        if self.best_metric is None:
            self.best_metric = current_metric
            if self.mode == 'auto':
                self.mode = 'min' if isinstance(self.monitor, str) and 'loss' in self.monitor.lower() else 'max'

        if self.mode == 'min':
            improved = current_metric < self.best_metric - self.min_delta
        else:
            improved = current_metric > self.best_metric + self.min_delta

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = [layer.weights for layer in model.layers if hasattr(layer, 'weights')]
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                for layer, best_weights in zip([layer for layer in model.layers if hasattr(layer, 'weights')],
                                               self.best_weights):
                    layer.weights = best_weights
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            return True

        return False

    def _get_monitor_value(self, logs):
        logs = logs or {}
        if isinstance(self.monitor, Metric):
            monitor_value = logs.get(self.monitor.name)
        else:
            monitor_value = logs.get(self.monitor)
        
        if monitor_value is None:
            if isinstance(logs, dict) and 'loss' in logs:
                monitor_value = logs['loss']
            elif isinstance(logs, (int, float)):
                monitor_value = logs
            else:
                raise ValueError(f"Monitored metric '{self.monitor}' is not available. "
                                 f"Available metrics are: {','.join(logs.keys())}")
        return monitor_value
