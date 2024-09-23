from neuralnetlib.metrics import Metric


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True,
                 start_from_epoch: int = 0, monitor: str = 'loss', mode: str = 'auto', baseline: float = None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.monitor = Metric(monitor).name if monitor != 'loss' else 'loss'
        self.mode = mode
        self.baseline = baseline
        self.best_weights = None
        self.best_metric = None
        self.patience_counter = 0
        self.stop_training = False

    def _get_monitor_value(self, logs):
        logs = logs or {}
        if isinstance(self.monitor, Metric):
            monitor_name = self.monitor.name
        else:
            monitor_name = self.monitor

        monitor_value = logs.get(monitor_name)
        if monitor_value is None:
            if isinstance(logs, dict) and 'loss' in logs:
                monitor_value = logs['loss']
            elif isinstance(logs, (int, float)):
                monitor_value = logs
            else:
                raise ValueError(f"Monitored metric '{monitor_name}' is not available. Available metrics are: {','.join(logs.keys())}")
        return monitor_value

    def on_epoch_end(self, model, epoch, logs):
        if epoch < self.start_from_epoch:
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
            print(f"\nEarly stopping after {epoch + 1} epochs.", end='')
            if self.start_from_epoch > 0:
                print(f" (with {self.start_from_epoch} epochs skipped),", end='')
            if self.restore_best_weights and self.best_weights is not None:
                for layer, best_weights in zip([layer for layer in model.layers if hasattr(layer, 'weights')],
                                               self.best_weights):
                    layer.weights = best_weights
            return True

        return False