import numpy as np

from neuralnetlib.metrics import Metric
from neuralnetlib.layers import Layer


class ModelWeightManager:
    # non trainable state that must be restored along with the parameters
    STATE_ATTRIBUTES = ('running_mean', 'running_var')

    @staticmethod
    def _get_layers(model) -> list:
        if hasattr(model, '_all_layers'):
            return model._all_layers()
        if hasattr(model, 'layers'):  # Sequential model
            return list(model.layers)
        if hasattr(model, 'encoder_layers') and hasattr(model, 'decoder_layers'):  # Autoencoder
            return list(model.encoder_layers) + list(model.decoder_layers)
        return []

    @staticmethod
    def _get_arrays(model) -> list[np.ndarray]:
        """All the arrays defining the state of the model, in a deterministic order."""
        arrays = []
        for layer in ModelWeightManager._get_layers(model):
            if hasattr(layer, 'get_trainable_parameters'):
                arrays.extend(param for _, param, _ in layer.get_trainable_parameters())
            for attribute in ModelWeightManager.STATE_ATTRIBUTES:
                value = getattr(layer, attribute, None)
                if isinstance(value, np.ndarray):
                    arrays.append(value)
        return arrays

    @staticmethod
    def get_model_weights(model) -> list[np.ndarray]:
        """Copy of all the parameters (and normalization statistics) of any model type."""
        return [array.copy() for array in ModelWeightManager._get_arrays(model)]

    @staticmethod
    def set_model_weights(model, params: list[np.ndarray]) -> None:
        """Restores, in place, the parameters saved by get_model_weights."""
        arrays = ModelWeightManager._get_arrays(model)
        if len(arrays) != len(params):
            raise ValueError("The saved weights do not match the architecture of the model.")
        for array, saved in zip(arrays, params):
            # in place: the optimizers and the layers keep referencing the same arrays
            array[...] = np.reshape(saved, array.shape)


class Callback:
    def on_train_begin(self, logs: dict | None = None) -> None:
        pass

    def on_train_end(self, logs: dict | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True,
                 start_from_epoch: int = 0, monitor: str = 'loss', mode: str = 'auto',
                 baseline: float | None = None) -> None:
        super().__init__()
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.restore_best_weights: bool = restore_best_weights
        self.start_from_epoch: int = start_from_epoch
        
        if monitor in ['loss', 'val_loss']:
            self.monitor = monitor
            self.is_metric = False
        else:
            try:
                self.monitor = Metric(monitor)
                self.is_metric = True
            except ValueError as e:
                if 'val_' in monitor:
                    try:
                        base_metric = monitor.replace('val_', '')
                        _ = Metric(base_metric)
                        self.monitor = monitor
                        self.is_metric = False
                    except ValueError:
                        raise ValueError(f"Invalid monitor metric: {monitor}") from e
                else:
                    raise ValueError(f"Invalid monitor metric: {monitor}") from e
        
        self.mode: str = mode
        self.baseline: float | None = baseline
        self.best_weights = None
        self.best_metric: float | None = None
        self.patience_counter: int = 0
        self.stop_training: bool = False
        self.weight_manager = ModelWeightManager()

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.patience_counter = 0
        self.best_metric = None
        self.best_weights = None
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> bool:
        logs = logs or {}
        model = logs.get('model')
        if epoch < self.start_from_epoch or model is None:
            return False

        current_metric = self._get_monitor_value(logs)

        # the mode must be known before comparing with the baseline
        if self.mode == 'auto':
            if isinstance(self.monitor, str):
                self.mode = 'min' if 'loss' in self.monitor.lower() else 'max'
            else:
                self.mode = 'min' if 'loss' in self.monitor.name.lower() else 'max'

        if self.baseline is not None and self.best_metric is None:
            if self.mode == 'min' and current_metric > self.baseline:
                print(f"\nEarly stopping: baseline {self.baseline} was not met.")
                return True
            if self.mode == 'max' and current_metric < self.baseline:
                print(f"\nEarly stopping: baseline {self.baseline} was not met.")
                return True

        if self.best_metric is None:
            # the first monitored epoch is the best one so far (its weights must be saved too)
            improved = True
        elif self.mode == 'min':
            improved = current_metric < self.best_metric - self.min_delta
        else:
            improved = current_metric > self.best_metric + self.min_delta

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = self.weight_manager.get_model_weights(model)
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.weight_manager.set_model_weights(model, self.best_weights)
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            return True

        return False

    def _get_monitor_value(self, logs: dict) -> float:
        logs = logs or {}
        
        if isinstance(self.monitor, str):
            monitor_value = logs.get(self.monitor)
        else:
            monitor_value = logs.get(self.monitor.name)

        if monitor_value is None:
            if isinstance(logs, dict) and 'loss' in logs:
                monitor_value = logs['loss']
            elif isinstance(logs, (int, float)):
                monitor_value = logs
            else:
                available_metrics = list(logs.keys())
                raise ValueError(
                    f"Monitored metric '{self.monitor if isinstance(self.monitor, str) else self.monitor.name}' "
                    f"is not available. Available metrics are: {', '.join(available_metrics)}"
                )
                
        return float(monitor_value)


class LearningRateScheduler(Callback):
    def __init__(self,
                 schedule,
                 initial_learning_rate: float = 0.01,
                 min_learning_rate: float = 1e-6,
                 schedule_params: dict = None,
                 verbose: bool = False) -> None:
        """Schedule can be a string or a callable function."""
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.verbose = verbose
        self.schedule_params = schedule_params or {}

        if isinstance(schedule, str):
            self.schedule = self._get_schedule_function(schedule)
        else:
            self.schedule = schedule

        self.current_learning_rate = initial_learning_rate

    def _step_decay(self, epoch: int, initial_learning_rate: float) -> float:
        drop_rate = self.schedule_params.get('drop_rate', 0.5)
        epochs_drop = self.schedule_params.get('epochs_drop', 10.0)

        learning_rate = initial_learning_rate * np.power(
            drop_rate, np.floor((1 + epoch) / epochs_drop))
        return max(learning_rate, self.min_learning_rate)

    def _exponential_decay(self, epoch: int, initial_learning_rate: float) -> float:
        decay_rate = self.schedule_params.get('decay_rate', 0.1)

        learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
        return max(learning_rate, self.min_learning_rate)

    def _cosine_decay(self, epoch: int, initial_learning_rate: float) -> float:
        total_epochs = self.schedule_params.get('total_epochs', 100)

        learning_rate = self.min_learning_rate + 0.5 * (initial_learning_rate - self.min_learning_rate) * \
            (1 + np.cos(np.pi * epoch / total_epochs))
        return max(learning_rate, self.min_learning_rate)

    def _warmup_cosine_decay(self, epoch: int, initial_learning_rate: float) -> float:
        warmup_epochs = self.schedule_params.get('warmup_epochs', 5)
        total_epochs = self.schedule_params.get('total_epochs', 100)

        if epoch < warmup_epochs:
            learning_rate = (epoch + 1) * initial_learning_rate / warmup_epochs
        else:
            learning_rate = self.min_learning_rate + 0.5 * (initial_learning_rate - self.min_learning_rate) * \
                (1 + np.cos(np.pi * (epoch - warmup_epochs) /
                 (total_epochs - warmup_epochs)))
        return max(learning_rate, self.min_learning_rate)

    def _cyclical(self, epoch: int, initial_learning_rate: float) -> float:
        step_size = self.schedule_params.get('step_size', 5)
        max_lr = self.schedule_params.get('max_lr', initial_learning_rate * 3)

        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        learning_rate = initial_learning_rate + \
            (max_lr - initial_learning_rate) * max(0, 1 - x)
        return max(learning_rate, self.min_learning_rate)

    def _get_schedule_function(self, schedule_name: str) -> callable:
        schedules = {
            'step': self._step_decay,
            'exponential': self._exponential_decay,
            'cosine': self._cosine_decay,
            'warmup_cosine': self._warmup_cosine_decay,
            'cyclical': self._cyclical
        }

        if schedule_name not in schedules:
            raise ValueError(
                f"Unknown schedule: {schedule_name}. Available schedules: {list(schedules.keys())}")

        return schedules[schedule_name]

    def _update_optimizer_learning_rate(self, model, new_lr: float) -> None:
        if hasattr(model, 'optimizer'):
            model.optimizer.learning_rate = new_lr
        elif hasattr(model, 'encoder_optimizer') and hasattr(model, 'decoder_optimizer'):
            model.encoder_optimizer.learning_rate = new_lr
            model.decoder_optimizer.learning_rate = new_lr
        elif hasattr(model, 'generator_optimizer') and hasattr(model, 'discriminator_optimizer'):
            model.generator_optimizer.learning_rate = new_lr
            model.discriminator_optimizer.learning_rate = new_lr

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        if not logs:
            return

        model = logs.get('model')
        if not model:
            return
        new_lr = self.schedule(epoch, self.initial_learning_rate)
        self._update_optimizer_learning_rate(model, new_lr)

        if self.verbose and new_lr != self.current_learning_rate:
            print(
                f'\nEpoch {epoch + 1}: Learning rate adjusted to {new_lr:.6f}')

        self.current_learning_rate = new_lr

    def on_train_begin(self, logs: dict = None) -> None:
        if self.verbose:
            print(f'Initial learning rate: {self.initial_learning_rate:.6f}')
