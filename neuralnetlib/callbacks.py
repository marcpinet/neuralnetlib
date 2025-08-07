import numpy as np

from neuralnetlib.metrics import Metric
from neuralnetlib.layers import Layer


class ModelWeightManager:
    @staticmethod
    def get_model_weights(model) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Extract weights and biases from any model type."""
        params = []

        def get_params_from_layer(layer):
            if hasattr(layer, 'weights'):
                weights = layer.weights.copy()
                bias = layer.bias.copy() if hasattr(layer, 'bias') else None
                return (weights, bias)
            return None

        def get_params_from_dense_layers(layers):
            layer_params = []
            for layer in layers:
                p = get_params_from_layer(layer)
                if p:
                    layer_params.append(p)
            return layer_params

        if hasattr(model, 'layers'):  # Sequential model
            for layer in model.layers:
                p = get_params_from_layer(layer)
                if p:
                    params.append(p)

        elif hasattr(model, 'encoder_layers') and hasattr(model, 'decoder_layers'):  # Autoencoder
            for layer in model.encoder_layers:
                p = get_params_from_layer(layer)
                if p:
                    params.append(p)
            for layer in model.decoder_layers:
                p = get_params_from_layer(layer)
                if p:
                    params.append(p)

        elif hasattr(model, 'src_embedding'):  # Transformer
            params.append(get_params_from_layer(model.src_embedding))
            params.append(get_params_from_layer(model.tgt_embedding))

            for encoder_layer in model.encoder_layers:
                params.extend(get_params_from_dense_layers([
                    encoder_layer.attention.query_dense,
                    encoder_layer.attention.key_dense,
                    encoder_layer.attention.value_dense,
                    encoder_layer.attention.output_dense,
                    encoder_layer.ffn.dense1,
                    encoder_layer.ffn.dense2
                ]))

            for decoder_layer in model.decoder_layers:
                params.extend(get_params_from_dense_layers([
                    decoder_layer.self_attention.query_dense,
                    decoder_layer.self_attention.key_dense,
                    decoder_layer.self_attention.value_dense,
                    decoder_layer.self_attention.output_dense,
                    decoder_layer.cross_attention.query_dense,
                    decoder_layer.cross_attention.key_dense,
                    decoder_layer.cross_attention.value_dense,
                    decoder_layer.cross_attention.output_dense,
                    decoder_layer.ffn.dense1,
                    decoder_layer.ffn.dense2
                ]))

            params.append(get_params_from_layer(model.output_layer))

        return [p for p in params if p is not None]

    @staticmethod
    def set_model_weights(model, params: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
        """Restore weights and biases to any model type."""
        param_idx = 0

        def set_params_for_layer(layer):
            nonlocal param_idx
            if hasattr(layer, 'weights'):
                if param_idx < len(params):
                    weights, bias = params[param_idx]
                    layer.weights = weights.copy()
                    if hasattr(layer, 'bias') and bias is not None:
                        layer.bias = bias.copy()
                    param_idx += 1
        
        def set_params_for_dense_layers(layers):
            for layer in layers:
                set_params_for_layer(layer)

        if hasattr(model, 'layers'):  # Sequential model
            for layer in model.layers:
                set_params_for_layer(layer)

        elif hasattr(model, 'encoder_layers') and hasattr(model, 'decoder_layers'):  # Autoencoder
            for layer in model.encoder_layers:
                set_params_for_layer(layer)
            for layer in model.decoder_layers:
                set_params_for_layer(layer)

        elif hasattr(model, 'src_embedding'): # Transformer
            set_params_for_layer(model.src_embedding)
            set_params_for_layer(model.tgt_embedding)

            for encoder_layer in model.encoder_layers:
                set_params_for_dense_layers([
                    encoder_layer.attention.query_dense,
                    encoder_layer.attention.key_dense,
                    encoder_layer.attention.value_dense,
                    encoder_layer.attention.output_dense,
                    encoder_layer.ffn.dense1,
                    encoder_layer.ffn.dense2
                ])

            for decoder_layer in model.decoder_layers:
                set_params_for_dense_layers([
                    decoder_layer.self_attention.query_dense,
                    decoder_layer.self_attention.key_dense,
                    decoder_layer.self_attention.value_dense,
                    decoder_layer.self_attention.output_dense,
                    decoder_layer.cross_attention.query_dense,
                    decoder_layer.cross_attention.key_dense,
                    decoder_layer.cross_attention.value_dense,
                    decoder_layer.cross_attention.output_dense,
                    decoder_layer.ffn.dense1,
                    decoder_layer.ffn.dense2
                ])
            
            set_params_for_layer(model.output_layer)


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
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> bool:
        logs = logs or {}
        model = logs.get('model')
        if epoch < self.start_from_epoch or model is None:
            return False

        current_metric = self._get_monitor_value(logs)
        
        if self.baseline is not None and self.best_metric is None:
            if self.mode == 'min' and current_metric > self.baseline:
                print(f"\nEarly stopping: baseline {self.baseline} was not met.")
                return True
            if self.mode == 'max' and current_metric < self.baseline:
                print(f"\nEarly stopping: baseline {self.baseline} was not met.")
                return True

        if self.best_metric is None:
            self.best_metric = current_metric
            if self.mode == 'auto':
                if isinstance(self.monitor, str):
                    self.mode = 'min' if 'loss' in self.monitor.lower() else 'max'
                else:
                    self.mode = 'min' if 'loss' in self.monitor.name.lower() else 'max'

        if self.mode == 'min':
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
