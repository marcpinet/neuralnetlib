import os
import inspect
import json
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from functools import lru_cache

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.callbacks import EarlyStopping
from neuralnetlib.layers import *
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy, CrossEntropyWithLabelSmoothing
from neuralnetlib.metrics import Metric
from neuralnetlib.optimizers import Optimizer
from neuralnetlib.preprocessing import PCA, pad_sequences, clip_gradients, SpectralNorm
from neuralnetlib.utils import shuffle, progress_bar, is_interactive, is_display_available, format_number, log_softmax, softmax, train_test_split, History, GradientDebugger, to_json_serializable


# layers whose 'activation' attribute is used internally, it must not be added to the model as a separate layer
_LAYERS_WITH_INTERNAL_ACTIVATION = (FeedForward, TransformerEncoderLayer, TransformerDecoderLayer)


@lru_cache(maxsize=None)
def _forward_accepts_training(layer_class: type) -> bool:
    try:
        return 'training' in inspect.signature(layer_class.forward_pass).parameters
    except (TypeError, ValueError):
        return False


def _layer_forward(layer, x, training: bool):
    """Forward pass of a layer, giving it the training flag when it uses one (dropout, normalization, RNNs...)."""
    if _forward_accepts_training(type(layer)):
        return layer.forward_pass(x, training=training)
    return layer.forward_pass(x)


def _activation_layer_for(layer):
    """Returns the Activation layer to add after a layer created with an `activation` argument (if any)."""
    if isinstance(layer, (Activation,) + _LAYERS_WITH_INTERNAL_ACTIVATION):
        return None

    activation_attr = getattr(layer, 'activation', getattr(
        layer, 'activation_function', None))
    if not activation_attr:
        return None
    if isinstance(activation_attr, str):
        return Activation.from_name(activation_attr)
    if isinstance(activation_attr, ActivationFunction):
        return Activation(activation_attr)
    if isinstance(activation_attr, Activation):
        return activation_attr
    raise ValueError(f"Invalid activation function: {activation_attr}")


def _derive_seed(random_state: int | None, *keys: int) -> int | None:
    """Deterministic seed derived from a base seed, so that different layers do not share the same seed."""
    if random_state is None:
        return None
    return int(np.random.SeedSequence([int(random_state), *[int(k) for k in keys]]).generate_state(1)[0])


def _assign_missing_seeds(layers: list, random_state: int | None, offset: int = 0) -> None:
    """Gives a seed to the layers without explicit seed (explicit seeds are kept)."""
    if random_state is None:
        return
    for i, layer in enumerate(layers):
        if hasattr(layer, 'random_state') and layer.random_state is None:
            layer.random_state = _derive_seed(random_state, offset + i)


def _update_layer_parameters(layer, key_prefix: str, optimizer: Optimizer, gradient_transform=None) -> None:
    """Updates (in place) all the trainable parameters of a layer, each parameter having its own optimizer state."""
    if not hasattr(layer, 'get_trainable_parameters'):
        return
    for name, param, grad in layer.get_trainable_parameters():
        if grad is None:
            continue
        if gradient_transform is not None:
            grad = gradient_transform(name, param, grad)
        optimizer.update(f"{key_prefix}.{name}", param, grad)


def _concatenate(arrays: list) -> np.ndarray:
    """Concatenates batches along the first axis (works for 1D arrays too, contrary to np.vstack)."""
    return np.concatenate([np.asarray(a) for a in arrays], axis=0)


def _pad_batch(array: np.ndarray, padded_size: int) -> np.ndarray:
    if array.shape[0] == padded_size:
        return array
    padding = np.zeros((padded_size - array.shape[0],) + array.shape[1:], dtype=array.dtype)
    return np.concatenate([array, padding], axis=0)


class BaseModel(ABC):
    def __init__(self, gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None):

        self.gradient_clip_threshold = gradient_clip_threshold
        self.enable_padding = enable_padding
        self.padding_size = padding_size
        # None means "not reproducible": a fixed default seed would make every shuffle (and dropout mask...) identical
        self.random_state = random_state

    def _create_batch_logs(self, batch: int, batch_size: int) -> dict:
        return {
            'batch': batch,
            'size': batch_size,
            'model': self
        }

    def _create_epoch_logs(self, epoch: int, logs: dict) -> dict:
        epoch_logs = {
            'epoch': epoch,
            'model': self
        }
        epoch_logs.update(logs)
        return epoch_logs

    def _clip(self, gradient: np.ndarray) -> np.ndarray:
        if gradient is None or not self.gradient_clip_threshold or self.gradient_clip_threshold <= 0:
            return gradient
        return clip_gradients(gradient, self.gradient_clip_threshold)

    def _pad_error(self, error: np.ndarray) -> np.ndarray:
        """The layers cached the padded batch (when padding is enabled), the error must have the same batch size."""
        original_size, padded_size = getattr(self, '_batch_sizes', (None, None))
        if padded_size is None or original_size == padded_size or error.shape[0] != original_size:
            return error
        return _pad_batch(error, padded_size)

    def _all_layers(self) -> list:
        """All the layers of the model (used by the callbacks to save/restore the weights)."""
        return []

    @abstractmethod
    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward_pass(self, error: np.ndarray):
        pass

    @abstractmethod
    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        pass

    @abstractmethod
    def compile(self, loss_function, optimizer, verbose: bool = False, metrics: list[Metric] = None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> tuple:
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, filename: str) -> 'BaseModel':
        pass


class Sequential(BaseModel):
    def __init__(self, gradient_clip_threshold: float = 1.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None,
                 n_classes: int | None = None):
        super().__init__(gradient_clip_threshold,
                         enable_padding, padding_size, random_state)
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.y_true = None
        self.predictions = None
        self._initialized = False
        self.n_classes = n_classes

    def __str__(self) -> str:
        model_summary = f'Sequential(gradient_clip_threshold={self.gradient_clip_threshold}, enable_padding={self.enable_padding}, padding_size={self.padding_size}, random_state={self.random_state})\n'
        model_summary += '-------------------------------------------------\n'
        for i, layer in enumerate(self.layers):
            model_summary += f'Layer {i + 1}: {str(layer)}\n'
        model_summary += '-------------------------------------------------\n'
        model_summary += f'Loss function: {str(self.loss_function)}\n'
        model_summary += f'Optimizer: {str(self.optimizer)}\n'
        model_summary += '-------------------------------------------------\n'
        return model_summary

    def summary(self):
        print(str(self))

    def _all_layers(self) -> list:
        return list(self.layers)

    def add(self, layer: Layer):
        if not self.layers:
            if not isinstance(layer, Input):
                raise ValueError("The first layer must be an Input layer.")
        else:
            if self.random_state is not None and hasattr(layer, 'random_state') and layer.random_state is None:
                layer.random_state = _derive_seed(self.random_state, len(self.layers))

            if isinstance(layer, Dense) and isinstance(self.layers[0], Input):
                layer.input_dim = self.layers[0].input_dim

            previous_layer = self.layers[-1]
            previous_type = type(previous_layer)
            current_type = type(layer)

            if previous_type in incompatibility_dict:
                if current_type in incompatibility_dict[previous_type]:
                    raise ValueError(
                        f"{current_type.__name__} layer cannot follow {previous_type.__name__} layer.")

            if isinstance(previous_layer, Attention) and isinstance(layer, Dense):
                previous_layer.return_sequences = False

        self.layers.append(layer)

        activation = _activation_layer_for(layer)
        if activation is not None:
            self.layers.append(activation)

    def compile(self, loss_function: LossFunction | str, optimizer: Optimizer | str, verbose: bool = False, metrics: list[Metric] = None):
        self.loss_function = loss_function if isinstance(loss_function, LossFunction) else LossFunction.from_name(
            loss_function)
        self.optimizer = optimizer if isinstance(
            optimizer, Optimizer) else Optimizer.from_name(optimizer)
        self.metrics = metrics
        if verbose:
            print(str(self))

    def forward_pass(self, X: np.ndarray, training: bool = True, labels: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X)
        if self.n_classes is not None and labels is not None:
            if len(X.shape) != 2:
                raise ValueError("Input shape must be (batch_size, features) for conditional models")

            if labels.ndim == 1:
                one_hot = np.zeros((labels.shape[0], self.n_classes))
                one_hot[np.arange(labels.shape[0]), labels] = 1
                labels = one_hot
            elif labels.shape[1] != self.n_classes:
                raise ValueError(f"Labels must have {self.n_classes} classes")

            X = np.concatenate([X, labels], axis=1)

        original_size = X.shape[0]
        if self.enable_padding:
            padded_size = (original_size + self.padding_size - 1) // self.padding_size * self.padding_size
            X = _pad_batch(X, padded_size)
        self._batch_sizes = (original_size, X.shape[0])

        for layer in self.layers:
            X = _layer_forward(layer, X, training)

        X = X[:original_size]

        self.predictions = X
        return X

    def _output_activation_error(self, layer: Activation, error: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. the input of the output activation. Softmax/sigmoid combined with a cross-entropy have a
        simple (and numerically stable) gradient, the other combinations go through the activation's backward pass."""
        activation_name = type(layer.activation_function).__name__

        if activation_name == "Softmax" and isinstance(self.loss_function, SparseCategoricalCrossentropy):
            labels = SparseCategoricalCrossentropy.to_class_indices(self.y_true, self.predictions)
            y_true_one_hot = np.zeros_like(self.predictions)
            np.put_along_axis(y_true_one_hot, labels[..., np.newaxis], 1, axis=-1)
            return self._pad_error(self.predictions - y_true_one_hot)
        if ((activation_name == "Softmax" and isinstance(self.loss_function, CategoricalCrossentropy)) or
                (activation_name == "Sigmoid" and isinstance(self.loss_function, BinaryCrossentropy))):
            return self._pad_error(self.predictions - np.reshape(self.y_true, self.predictions.shape))

        return layer.backward_pass(self._pad_error(error))

    def backward_pass(self, error: np.ndarray, gan: bool = False, compute_only: bool = False) -> np.ndarray:

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0 and isinstance(layer, Activation):
                if gan:
                    error = layer.backward_pass(self._pad_error(error))
                else:
                    error = self._output_activation_error(layer, error)
            else:
                if i == 0:
                    error = self._pad_error(error)
                error = self._clip(error)
                error = layer.backward_pass(error)

            if compute_only:
                continue

            layer_idx = len(self.layers) - 1 - i
            _update_layer_parameters(layer, str(layer_idx), self.optimizer,
                                     lambda name, param, grad: self._clip(grad))

        original_size = getattr(self, '_batch_sizes', (None, None))[0]
        if error is not None and original_size is not None:
            error = error[:original_size]
        return error

    def _format_targets(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """1D targets are reshaped to match the (batch_size, 1) predictions (but class indices are kept as is)."""
        y = np.asarray(y)
        if isinstance(self.loss_function, SparseCategoricalCrossentropy):
            return y
        if y.ndim == predictions.ndim - 1 and predictions.shape[-1] == 1:
            return y.reshape(predictions.shape)
        return y

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        self.predictions = self.forward_pass(x_batch)
        y_batch = self._format_targets(y_batch, self.predictions)
        self.y_true = y_batch
        predictions = self.predictions.copy()
        loss = float(self.loss_function(y_batch, predictions))
        error = self.loss_function.derivative(y_batch, predictions)

        if error.ndim == 1:
            error = error[:, None]
        elif isinstance(self.layers[-1], (LSTM, Bidirectional, GRU)) and self.layers[-1].return_sequences:
            error = error.reshape(error.shape[0], error.shape[1], -1)

        self.backward_pass(error)
        return loss

    def _process_metrics(self, metrics: list | None, history: History, with_validation: bool) -> list | None:
        if metrics is None:
            return None
        processed_metrics = []
        for metric in metrics:
            if metric == 'val_loss':
                continue
            try:
                metric_obj = Metric(metric)
            except ValueError as e:
                raise ValueError(f"Invalid metric: {metric}") from e
            processed_metrics.append(metric_obj)
            history[metric_obj.name] = []
            if with_validation:
                history[f'val_{metric_obj.name}'] = []
        return processed_metrics

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
                epochs: int,
                batch_size: int | None = None,
                verbose: bool = True,
                metrics: list | None = None,
                random_state: int | None = None,
                validation_data: tuple | None = None,
                validation_split: float | None = None,
                callbacks: list = [],
                plot_decision_boundary: bool = False) -> dict:
        """
        Fit the model to the training data.

        Args:
            x_train: Training data
            y_train: Training labels
            epochs: Number of epochs to train the model
            batch_size: Number of samples per gradient update
            verbose: Whether to print training progress
            metrics: List of metric to evaluate the model
            random_state: Random seed for shuffling the data
            validation_data: Tuple of validation data and labels
            callbacks: List of callback objects (e.g., EarlyStopping)
            plot_decision_boundary: Whether to plot the decision boundary

        Returns:
            Dictionary containing the training history of metrics (loss and any other metrics)
        """

        if hasattr(self, 'metrics') and self.metrics is not None:
            metrics = self.metrics

        seed = random_state if random_state is not None else self.random_state

        history = History({
            'loss': [],
            'val_loss': []
        })

        if validation_split is not None and validation_data is not None:
            raise ValueError("Cannot specify both validation_data and validation_split")
        elif validation_split is not None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=validation_split,
                random_state=seed
            )
            validation_data = (x_val, y_val)

        if plot_decision_boundary and not is_interactive() and not is_display_available():
            raise ValueError("Cannot display the plot. Please run the script in an environment with a display.")
        if plot_decision_boundary:
            # matplotlib is only needed for the plots
            import matplotlib.pyplot as plt

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        _assign_missing_seeds(self.layers, seed)

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU, Unidirectional)) for layer in self.layers)
        has_embedding = any(isinstance(layer, Embedding) for layer in self.layers)

        if has_lstm_or_gru and not has_embedding:
            if len(x_train.shape) != 3:
                raise ValueError(
                    "Input data must be 3D (batch_size, time_steps, features) for LSTM/GRU layers without Embedding"
                )
        elif has_embedding:
            if len(x_train.shape) != 2:
                raise ValueError(
                    "Input data must be 2D (batch_size, sequence_length) when using Embedding layer"
                )

        if validation_data is not None:
            x_val, y_val = validation_data
            x_val = np.array(x_val)
            y_val = np.array(y_val)

        metrics = self._process_metrics(metrics, history, validation_data is not None)

        for layer in self.layers:
            if isinstance(layer, TextVectorization):
                layer.adapt(x_train)
                break

        callbacks = callbacks if callbacks is not None else []

        logs = {
            'model': self,
            'params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': [m.name for m in (metrics or [])],
                'validation': validation_data is not None,
            }
        }

        for callback in callbacks:
            callback.on_train_begin(logs)

        # a single generator, so that the data is shuffled differently at each epoch (and reproducibly)
        shuffle_rng = np.random.default_rng(seed)

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()
                permutation = shuffle_rng.permutation(x_train.shape[0])
                x_train_shuffled = x_train[permutation]
                y_train_shuffled = y_train[permutation]

                error = 0
                predictions_list = []
                y_true_list = []
                val_loss = val_predictions = None

                if batch_size is not None:
                    num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)

                    for j in range(0, x_train.shape[0], batch_size):
                        batch_index = j // batch_size

                        x_batch = x_train_shuffled[j:j + batch_size]
                        y_batch = y_train_shuffled[j:j + batch_size]

                        batch_logs = {
                            'batch': batch_index,
                            'size': len(x_batch),
                            'model': self
                        }
                        for callback in callbacks:
                            callback.on_batch_begin(batch_index, batch_logs)

                        batch_error = self.train_on_batch(x_batch, y_batch)
                        error += batch_error
                        # kept only for the metrics: the predictions of a whole epoch can be very large
                        if metrics is not None:
                            predictions_list.append(self.predictions)
                            y_true_list.append(self.y_true)

                        batch_logs.update({'loss': batch_error})

                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(predictions_list[-1], y_true_list[-1])
                                batch_metrics[metric.name] = batch_metric_value
                            batch_logs.update(batch_metrics)

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        val_metrics_str = ''
                        if validation_data is not None and batch_index == num_batches - 1:
                            val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                            val_metrics_str = f'val_loss: {format_number(val_loss)} - '
                            if metrics is not None:
                                for metric in metrics:
                                    val_metric = metric(val_predictions, y_val)
                                    val_metrics_str += f'val_{metric.name}: {format_number(val_metric)} - '

                        if verbose:
                            current_loss = error / (batch_index + 1)
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metric_value = metric(
                                        _concatenate(predictions_list),
                                        _concatenate(y_true_list)
                                    )
                                    metrics_str += f'{metric.name}: {format_number(metric_value)} - '

                            progress_message = (
                                f'Epoch {epoch + 1}/{epochs} - {time.time() - start_time:.2f}s - '
                                f'loss: {format_number(current_loss)} - '
                                f'{metrics_str}{val_metrics_str}'.rstrip(' -')
                            )

                            progress_bar(
                                batch_index + 1,
                                num_batches,
                                message=progress_message
                            )

                    error /= num_batches

                else:
                    error = self.train_on_batch(x_train_shuffled, y_train_shuffled)
                    # kept only for the metrics: the predictions of a whole epoch can be very large
                    if metrics is not None:
                        predictions_list.append(self.predictions)
                        y_true_list.append(self.y_true)

                    val_metrics_str = ''
                    if validation_data is not None:
                        val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                        val_metrics_str = f'val_loss: {format_number(val_loss)} - '
                        if metrics is not None:
                            for metric in metrics:
                                val_metric = metric(val_predictions, y_val)
                                val_metrics_str += f'val_{metric.name}: {format_number(val_metric)} - '

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(
                                    _concatenate(predictions_list),
                                    _concatenate(y_true_list)
                                )
                                metrics_str += f'{metric.name}: {format_number(metric_value)} - '

                        progress_message = (
                            f'Epoch {epoch + 1}/{epochs} - {time.time() - start_time:.2f}s - '
                            f'loss: {format_number(error)} - '
                            f'{metrics_str}{val_metrics_str}'.rstrip(' -')
                        )

                        progress_bar(1, 1, message=progress_message)

                history['loss'].append(error)

                epoch_logs.update({
                    'loss': error,
                    'time': time.time() - start_time
                })

                if metrics is not None:
                    for metric in metrics:
                        metric_value = metric(
                            _concatenate(predictions_list),
                            _concatenate(y_true_list)
                        )
                        history[metric.name].append(metric_value)
                        epoch_logs[metric.name] = metric_value

                if validation_data is not None:
                    if val_loss is None:
                        val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

                    if metrics is not None:
                        for metric in metrics:
                            val_metric = metric(val_predictions, y_val)
                            history[f'val_{metric.name}'].append(val_metric)
                            epoch_logs[f'val_{metric.name}'] = val_metric

                stop_training = False
                for callback in callbacks:
                    if callback.on_epoch_end(epoch, epoch_logs):
                        stop_training = True
                        break

                if verbose:
                    print()

                if plot_decision_boundary:
                    self.__update_plot(
                        epoch, x_train, y_train,
                        seed
                    )
                    plt.pause(0.1)

                if stop_training:
                    break

            if plot_decision_boundary:
                plt.show(block=True)

        finally:
            final_logs = {
                'model': self,
                'history': history
            }
            for callback in callbacks:
                callback.on_train_end(final_logs)

            if verbose:
                print()

        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> tuple:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        if batch_size is None:
            batch_size = len(x_test)

        total_loss = 0

        predictions_list = []

        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]

            batch_predictions = self.forward_pass(batch_x, training=False)
            batch_y = self._format_targets(batch_y, batch_predictions)
            batch_loss = self.loss_function(batch_y, batch_predictions)

            # weighted by the batch size, the last batch can be smaller
            total_loss += batch_loss * len(batch_x)
            predictions_list.append(batch_predictions)

            for layer in self.layers:
                if hasattr(layer, 'reset_cache'):
                    layer.reset_cache()

        avg_loss = float(total_loss / len(x_test))

        all_predictions = _concatenate(predictions_list)
        predictions_list = None

        try:
            frame = inspect.currentframe()
            calling_frame = frame.f_back
            code = calling_frame.f_code
            if 'single' in code.co_varnames:
                return avg_loss
        except:
            pass
        finally:
            del frame  # to avoid leaking references

        return avg_loss, all_predictions

    def predict(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        X = np.array(X)
        predictions = self.forward_pass(X, training=False)

        if not np.isclose(temperature, 1.0, rtol=1e-09, atol=1e-09):
            if isinstance(predictions, np.ndarray):
                predictions = np.clip(predictions, 1e-7, 1.0)
                log_preds = np.log(predictions)
                scaled_log_preds = log_preds / temperature
                predictions = np.exp(scaled_log_preds)
                predictions /= np.sum(predictions, axis=-1, keepdims=True)

        return predictions

    def generate_sequence(self,
                          sequence_start: np.ndarray,
                          max_length: int,
                          stop_token: int | None = None,
                          min_length: int | None = None,
                          temperature: float = 1.0) -> np.ndarray:

        current_sequence = sequence_start.copy()

        # the generator is kept between the calls: each call generates a new sequence (reproducibly if seeded)
        if getattr(self, '_generation_rng', None) is None:
            self._generation_rng = np.random.default_rng(self.random_state)
        rng = self._generation_rng

        for _ in range(max_length - sequence_start.shape[1]):
            # cuz we already apply temperature in this method
            predictions = self.predict(current_sequence)

            if predictions.ndim == 3:
                next_token_probs = predictions[:, -1, :]
            else:
                next_token_probs = predictions

            next_token_probs = np.array(next_token_probs, dtype=np.float64)

            if not np.isclose(temperature, 1.0, rtol=1e-09, atol=1e-09):
                next_token_probs = np.clip(next_token_probs, 1e-7, 1.0)
                log_probs = np.log(next_token_probs)
                scaled_log_probs = log_probs / temperature
                next_token_probs = np.exp(scaled_log_probs)
                next_token_probs /= np.sum(next_token_probs,
                                           axis=-1, keepdims=True)

            if min_length is not None and current_sequence.shape[1] < min_length:
                if stop_token is not None:
                    next_token_probs[:, stop_token] = 0
                    next_token_probs /= np.sum(next_token_probs,
                                               axis=-1, keepdims=True)

            next_tokens = []
            for probs in next_token_probs:
                if np.isnan(probs).any() or np.sum(probs) == 0:
                    next_token = rng.integers(0, probs.shape[0])
                else:
                    probs = probs / np.sum(probs)
                    next_token = rng.choice(probs.shape[0], p=probs)
                next_tokens.append(next_token)

            next_tokens = np.array(next_tokens)

            if stop_token is not None:
                if min_length is None or current_sequence.shape[1] >= min_length:
                    if np.all(next_tokens == stop_token):
                        break

            current_sequence = np.hstack(
                [current_sequence, next_tokens.reshape(-1, 1)])

        return current_sequence

    def save(self, filename: str):
        model_state = {
            'type': 'Sequential',
            'layers': [],
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state,
            'n_classes': self.n_classes
        }

        for layer in self.layers:
            model_state['layers'].append(layer.get_config())

        if self.loss_function:
            model_state['loss_function'] = self.loss_function.get_config()
        if self.optimizer:
            model_state['optimizer'] = self.optimizer.get_config()

        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4, default=to_json_serializable)

        return model_state

    @classmethod
    def load(cls, filename: str) -> 'Sequential':
        with open(filename, 'r') as f:
            model_state = json.load(f)

        model = cls()

        model_attributes = vars(model)

        for param, value in model_state.items():
            if param in model_attributes:
                setattr(model, param, value)

        model.layers = [
            Layer.from_config(layer_config) for layer_config in model_state.get('layers', [])
        ]

        if 'loss_function' in model_state:
            model.loss_function = LossFunction.from_config(
                model_state['loss_function'])
        if 'optimizer' in model_state:
            model.optimizer = Optimizer.from_config(model_state['optimizer'])

        return model

    def __update_plot(self, epoch: int, x_train: np.ndarray, y_train: np.ndarray, random_state: int | None) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        if not plt.fignum_exists(1):
            if matplotlib.get_backend() != "TkAgg":
                matplotlib.use("TkAgg")
                plt.ion()

            fig, ax = plt.subplots(figsize=(8, 6), num=1)
            pca = PCA(n_components=2, random_state=random_state)
            x_train_2d = pca.fit_transform(x_train)
            fig.pca = pca
        else:
            fig = plt.gcf()
            ax = fig.axes[0]
            pca = fig.pca
            x_train_2d = pca.transform(x_train)

        x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
        y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        if y_train.ndim > 1 and y_train.shape[1] > 1:
            y_train_encoded = np.argmax(y_train, axis=1)
        else:
            y_train_encoded = y_train.ravel()

        ax.clear()

        scatter = ax.scatter(
            x_train_2d[:, 0], x_train_2d[:, 1], c=y_train_encoded, cmap='viridis', alpha=0.7)

        labels = np.unique(y_train_encoded)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)),
                       label=f'Class {label}', markersize=8) for label in labels]
        ax.legend(handles=handles, title='Classes')

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(pca.inverse_transform(grid_points))
        if Z.shape[1] > 1:  # Multiclass classification
            Z = np.argmax(Z, axis=1).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu,
                        levels=np.arange(Z.max() + 1))
        else:  # Binary classification
            Z = (Z > 0.5).astype(int).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu, levels=1)

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title(f"Decision Boundary (Epoch {epoch + 1})")

        fig.canvas.draw()
        plt.pause(0.1)


class Autoencoder(BaseModel):
    def __init__(self,
                 encoder_layers: list = None,
                 decoder_layers: list = None,
                 gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None,
                 skip_connections: bool = False,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.0,
                 variational: bool = False):
        super().__init__(gradient_clip_threshold,
                         enable_padding, padding_size, random_state)

        self.encoder_layers = encoder_layers if encoder_layers is not None else []
        self.decoder_layers = decoder_layers if decoder_layers is not None else []

        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.encoder_loss = None
        self.decoder_loss = None

        self.y_true = None
        self.predictions = None
        self.latent_space = None
        self.latent_mean = None
        self.latent_log_var = None
        self.skip_connections = skip_connections

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.variational = variational
        self.skip_cache = {}

        self.epsilon = 1e-7
        # weight of the KL divergence in the loss of a variational autoencoder
        self.beta = 0.01

        self.latent_dim = None

    def _all_layers(self) -> list:
        return list(self.encoder_layers) + list(self.decoder_layers)

    def _get_rng(self) -> np.random.Generator:
        # a single generator, so that the noise of the reparameterization trick changes at every step
        if getattr(self, '_rng', None) is None:
            self._rng = np.random.default_rng(self.random_state)
        return self._rng

    def _calculate_kl_divergence(self):
        if not self.variational:
            return 0.0
        kl_loss = -0.5 * np.mean(
            1 + self.latent_log_var -
            np.square(self.latent_mean) - np.exp(self.latent_log_var)
        )
        return kl_loss

    def _reparameterize(self, training: bool = True):
        if not self.variational:
            return self.latent_space
        if not training:
            # the mean of the latent distribution is used for inference
            self._noise = np.zeros_like(self.latent_mean)
            return self.latent_mean
        self._noise = self._get_rng().normal(size=self.latent_mean.shape)
        return self.latent_mean + np.exp(0.5 * self.latent_log_var) * self._noise

    def _trainable_weights(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for name, param, _ in layer.get_trainable_parameters():
                if name == 'weights' or name.endswith('.weights'):
                    yield param

    def _calculate_regularization(self):
        reg_loss = 0.0

        for weights in self._trainable_weights():
            if self.l1_reg > 0:
                reg_loss += self.l1_reg * np.sum(np.abs(weights))
            if self.l2_reg > 0:
                reg_loss += self.l2_reg * np.sum(np.square(weights))

        return reg_loss

    def _latent_penalties(self) -> tuple:
        """Small penalties keeping the latent space well distributed.
        Returns (latent_l2, distribution_penalty, their gradient w.r.t. the latent space)."""
        latent = self.latent_space
        l2_factor = 0.0001
        distribution_factor = 0.0001
        if self.skip_connections:
            l2_factor *= 0.1
            distribution_factor *= 0.1

        latent_l2 = l2_factor * np.mean(np.square(latent))
        gradient = l2_factor * 2 * latent / latent.size

        latent_mean = np.mean(latent, axis=0)
        latent_std = np.std(latent, axis=0)
        distribution_penalty = distribution_factor * np.mean(np.abs(latent_std - 1.0))
        gradient = gradient + distribution_factor * np.sign(latent_std - 1.0) * (latent - latent_mean) / \
            (latent.shape[0] * (latent_std + 1e-8) * latent_std.size)

        return latent_l2, distribution_penalty, gradient

    def _compute_loss(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        reconstruction_loss = self.decoder_loss(y_true, predictions)
        regularization_loss = self._calculate_regularization()
        kl_loss = self._calculate_kl_divergence() if self.variational else 0
        latent_l2, distribution_penalty, _ = self._latent_penalties()

        return float(reconstruction_loss + regularization_loss +
                     latent_l2 + distribution_penalty + self.beta * kl_loss)

    def _apply_skip_connection(self, current_output: np.ndarray, decoder_idx: int) -> np.ndarray:
        if not self.skip_connections:
            return current_output

        encoder_idx = len(self.encoder_layers) - decoder_idx - 2

        if encoder_idx < 0 or encoder_idx >= len(self.encoder_layers):
            return current_output

        encoder_output = self.skip_cache.get(encoder_idx)
        if encoder_output is None:
            return current_output

        if encoder_output.shape == current_output.shape:
            alpha = 0.7
            return alpha * current_output + (1 - alpha) * encoder_output
        else:
            try:
                target_shape = current_output.shape
                if len(encoder_output.shape) == len(target_shape):
                    reshaped_output = np.resize(encoder_output, target_shape)
                    alpha = 0.7
                    return alpha * current_output + (1 - alpha) * reshaped_output
            except ValueError:
                pass

            return current_output

    def add_encoder_layer(self, layer: Layer):
        if not self.encoder_layers:
            if not isinstance(layer, Input):
                raise ValueError(
                    "The first encoder layer must be an Input layer.")
        else:
            previous_layer = self.encoder_layers[-1]
            previous_type = type(previous_layer)
            current_type = type(layer)

            if previous_type in incompatibility_dict:
                if current_type in incompatibility_dict[previous_type]:
                    raise ValueError(
                        f"{current_type.__name__} layer cannot follow {previous_type.__name__} layer.")

            _assign_missing_seeds([layer], self.random_state, len(self.encoder_layers))

        self.encoder_layers.append(layer)

        activation = _activation_layer_for(layer)
        if activation is not None:
            self.encoder_layers.append(activation)

    def add_decoder_layer(self, layer: Layer):
        if self.decoder_layers:
            previous_layer = self.decoder_layers[-1]
            previous_type = type(previous_layer)
            current_type = type(layer)

            if previous_type in incompatibility_dict:
                if current_type in incompatibility_dict[previous_type]:
                    raise ValueError(
                        f"{current_type.__name__} layer cannot follow {previous_type.__name__} layer.")

        _assign_missing_seeds([layer], self.random_state, 1000 + len(self.decoder_layers))

        self.decoder_layers.append(layer)

        activation = _activation_layer_for(layer)
        if activation is not None:
            self.decoder_layers.append(activation)

    def compile(self,
                encoder_loss: LossFunction | str = None,
                decoder_loss: LossFunction | str = None,
                encoder_optimizer: Optimizer | str = None,
                decoder_optimizer: Optimizer | str = None,
                verbose: bool = False,
                metrics: list[Metric] = None):

        if encoder_loss is None:
            encoder_loss = decoder_loss
        if decoder_loss is None:
            decoder_loss = encoder_loss
        if encoder_optimizer is None:
            encoder_optimizer = decoder_optimizer
        if decoder_optimizer is None:
            decoder_optimizer = encoder_optimizer

        if encoder_loss is None or encoder_optimizer is None:
            raise ValueError(
                "At least one loss and optimizer must be specified")

        self.encoder_loss = encoder_loss if isinstance(
            encoder_loss, LossFunction) else LossFunction.from_name(encoder_loss)
        self.decoder_loss = decoder_loss if isinstance(
            decoder_loss, LossFunction) else LossFunction.from_name(decoder_loss)
        self.encoder_optimizer = encoder_optimizer if isinstance(
            encoder_optimizer, Optimizer) else Optimizer.from_name(encoder_optimizer)
        self.decoder_optimizer = decoder_optimizer if isinstance(
            decoder_optimizer, Optimizer) else Optimizer.from_name(decoder_optimizer)

        self.metrics = metrics

        if verbose:
            print(str(self))

        # the latent dimension is given by the last layer with units of the encoder (Activation, BatchNormalization...
        # may follow it). For a variational autoencoder, the encoder outputs the mean and the log variance.
        self.latent_dim = None
        for layer in reversed(self.encoder_layers):
            if hasattr(layer, 'units'):
                self.latent_dim = layer.units // 2 if self.variational else layer.units
                break

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        X = np.asarray(X)
        original_size = X.shape[0]
        if self.enable_padding:
            padded_size = (original_size + self.padding_size - 1) // self.padding_size * self.padding_size
            X = _pad_batch(X, padded_size)
        self._batch_sizes = (original_size, X.shape[0])

        self.encoder_activations = []
        self.decoder_activations = []
        self.skip_cache = {}
        self._skip_links = {}

        # Encoder forward pass
        encoded = X
        for i, layer in enumerate(self.encoder_layers):
            encoded = _layer_forward(layer, encoded, training)
            self.encoder_activations.append(encoded)

            if self.skip_connections and isinstance(layer, Dense):
                self.skip_cache[layer.units] = (i, encoded)

        self.encoded = encoded

        if self.variational:
            latent_dim = encoded.shape[-1] // 2
            self.latent_dim = latent_dim
            self.latent_mean = encoded[:, :latent_dim]
            self.latent_log_var = encoded[:, latent_dim:]
            self.latent_space = self._reparameterize(training)
        else:
            self.latent_space = encoded

        # Decoder forward pass (from the sampled latent vector for a variational autoencoder)
        decoded = self.latent_space

        for j, layer in enumerate(self.decoder_layers):
            decoded = _layer_forward(layer, decoded, training)
            if self.skip_connections and isinstance(layer, Dense):
                skip = self.skip_cache.get(layer.units)
                if skip is not None and skip[1].shape == decoded.shape:
                    encoder_idx, skip_connection = skip
                    scale_factor = 1.0 / np.sqrt(layer.units)
                    decoded = decoded + scale_factor * skip_connection
                    self._skip_links[j] = (encoder_idx, scale_factor)

            self.decoder_activations.append(decoded)

        return decoded[:original_size]

    def _clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        if gradient is None:
            return None

        if self.gradient_clip_threshold > 0:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > self.gradient_clip_threshold:
                gradient = gradient * \
                    (self.gradient_clip_threshold / grad_norm)

            gradient = np.clip(gradient, -10, 10)

            batch_std = np.std(gradient) + 1e-8
            gradient = gradient / batch_std

        return gradient

    def _parameter_gradient(self, name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if name == 'weights' or name.endswith('.weights'):
            if self.l1_reg > 0:
                grad = grad + self.l1_reg * np.sign(param)
            if self.l2_reg > 0:
                grad = grad + 2 * self.l2_reg * param
        return self._clip_gradients(grad)

    def _output_activation_error(self, layer: Activation, error: np.ndarray) -> np.ndarray:
        activation_name = type(layer.activation_function).__name__
        if ((activation_name == "Softmax" and isinstance(self.decoder_loss, CategoricalCrossentropy)) or
                (activation_name == "Sigmoid" and isinstance(self.decoder_loss, BinaryCrossentropy))):
            # gradient w.r.t. the input of the activation
            return self._pad_error(self.predictions - np.reshape(self.y_true, self.predictions.shape))
        return layer.backward_pass(self._pad_error(error))

    def backward_pass(self, error: np.ndarray):
        # the encoder and the decoder may share the same optimizer: their parameters must have different keys
        skip_gradients = {}

        # Decoder backward pass
        for i, layer in enumerate(reversed(self.decoder_layers)):
            layer_idx = len(self.decoder_layers) - 1 - i

            if i == 0 and isinstance(layer, Activation):
                error = self._output_activation_error(layer, error)
            else:
                if i == 0:
                    error = self._pad_error(error)
                if layer_idx in self._skip_links:
                    # the output of this layer was summed with an encoder output
                    encoder_idx, scale_factor = self._skip_links[layer_idx]
                    skip_gradients[encoder_idx] = skip_gradients.get(encoder_idx, 0) + scale_factor * error
                error = self._clip_gradients(error)
                error = layer.backward_pass(error)

            _update_layer_parameters(layer, f"decoder.{layer_idx}", self.decoder_optimizer, self._parameter_gradient)

        # Latent space: regularization penalties, reparameterization trick and KL divergence
        error = error + self._latent_penalties()[2]

        if self.variational:
            n = self.latent_mean.size
            d_mean = error + self.beta * self.latent_mean / n
            d_log_var = error * self._noise * 0.5 * np.exp(0.5 * self.latent_log_var) + \
                self.beta * 0.5 * (np.exp(self.latent_log_var) - 1) / n
            error = np.concatenate([d_mean, d_log_var], axis=-1)

        # Encoder backward pass
        for i, layer in enumerate(reversed(self.encoder_layers)):
            layer_idx = len(self.encoder_layers) - 1 - i

            if layer_idx in skip_gradients:
                error = error + skip_gradients[layer_idx]

            error = self._clip_gradients(error)
            error = layer.backward_pass(error)

            _update_layer_parameters(layer, f"encoder.{layer_idx}", self.encoder_optimizer, self._parameter_gradient)

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray = None) -> float:
        if y_batch is None:
            y_batch = x_batch

        self.predictions = self.forward_pass(x_batch, training=True)
        self.y_true = np.asarray(y_batch)

        total_loss = self._compute_loss(self.y_true, self.predictions)

        error = self.decoder_loss.derivative(self.y_true, self.predictions)
        if error.ndim == 1:
            error = error[:, None]
        elif isinstance(self.decoder_layers[-1], (LSTM, Bidirectional, GRU)) and self.decoder_layers[-1].return_sequences:
            error = error.reshape(error.shape[0], error.shape[1], -1)

        self.backward_pass(error)
        return total_loss

    def predict(self, X: np.ndarray, output_latent: bool = False, temperature: float = 1.0) -> np.ndarray:
        X = np.array(X)
        # same computation as during training (skip connections included), in inference mode
        decoded = self.forward_pass(X, training=False)

        if output_latent:
            return self.encoded[:X.shape[0]]

        if not np.isclose(temperature, 1.0, rtol=1e-09, atol=1e-09):
            if isinstance(decoded, np.ndarray):
                decoded = np.clip(decoded, 1e-7, 1.0)
                log_preds = np.log(decoded)
                scaled_log_preds = log_preds / temperature
                decoded = np.exp(scaled_log_preds)
                decoded /= np.sum(decoded, axis=-1, keepdims=True)

        return decoded

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray = None, batch_size: int = 32) -> tuple:
        if y_test is None:
            y_test = x_test
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        if batch_size is None:
            batch_size = len(x_test)

        total_loss = 0
        predictions_list = []

        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]

            batch_predictions = self.forward_pass(batch_x, training=False)
            # same loss as the one minimized during training
            batch_loss = self._compute_loss(batch_y, batch_predictions)

            total_loss += batch_loss * len(batch_x)
            predictions_list.append(batch_predictions)

        avg_loss = float(total_loss / len(x_test))
        all_predictions = _concatenate(predictions_list)

        return avg_loss, all_predictions

    @classmethod
    def load(cls, filename: str) -> 'Autoencoder':
        with open(filename, 'r') as f:
            model_state = json.load(f)

        model = cls()

        model_attributes = vars(model)

        for param, value in model_state.items():
            if param in model_attributes:
                setattr(model, param, value)

        model.encoder_layers = [
            Layer.from_config(layer_config) for layer_config in model_state.get('encoder_layers', [])
        ]
        model.decoder_layers = [
            Layer.from_config(layer_config) for layer_config in model_state.get('decoder_layers', [])
        ]

        if 'encoder_loss' in model_state:
            model.encoder_loss = LossFunction.from_config(
                model_state['encoder_loss'])
        if 'decoder_loss' in model_state:
            model.decoder_loss = LossFunction.from_config(
                model_state['decoder_loss'])
        if 'encoder_optimizer' in model_state:
            model.encoder_optimizer = Optimizer.from_config(
                model_state['encoder_optimizer'])
        if 'decoder_optimizer' in model_state:
            model.decoder_optimizer = Optimizer.from_config(
                model_state['decoder_optimizer'])

        return model

    def save(self, filename: str):
        model_state = {
            'type': 'Autoencoder',
            'encoder_layers': [],
            'decoder_layers': [],
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state,
            'skip_connections': self.skip_connections,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'variational': self.variational,
            'beta': self.beta,
            'latent_dim': self.latent_dim
        }

        for layer in self.encoder_layers:
            model_state['encoder_layers'].append(layer.get_config())
        for layer in self.decoder_layers:
            model_state['decoder_layers'].append(layer.get_config())

        if self.encoder_loss:
            model_state['encoder_loss'] = self.encoder_loss.get_config()
        if self.decoder_loss:
            model_state['decoder_loss'] = self.decoder_loss.get_config()
        if self.encoder_optimizer:
            model_state['encoder_optimizer'] = self.encoder_optimizer.get_config()
        if self.decoder_optimizer:
            model_state['decoder_optimizer'] = self.decoder_optimizer.get_config()

        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4, default=to_json_serializable)

    def __str__(self) -> str:
        model_summary = f'Autoencoder(gradient_clip_threshold={self.gradient_clip_threshold}, ' \
            f'enable_padding={self.enable_padding}, padding_size={self.padding_size}, random_state={self.random_state}, ' \
            f'skip_connections={self.skip_connections}, l1_reg={self.l1_reg}, l2_reg={self.l2_reg})\n'
        model_summary += '-------------------------------------------------\n'
        model_summary += 'Encoder:\n'
        for i, layer in enumerate(self.encoder_layers):
            model_summary += f'Layer {i + 1}: {str(layer)}\n'
        model_summary += '-------------------------------------------------\n'
        model_summary += 'Decoder:\n'
        for i, layer in enumerate(self.decoder_layers):
            model_summary += f'Layer {i + 1}: {str(layer)}\n'
        model_summary += '-------------------------------------------------\n'
        model_summary += f'Encoder loss function: {str(self.encoder_loss)}\n'
        model_summary += f'Decoder loss function: {str(self.decoder_loss)}\n'
        model_summary += f'Encoder optimizer: {str(self.encoder_optimizer)}\n'
        model_summary += f'Decoder optimizer: {str(self.decoder_optimizer)}\n'
        model_summary += '-------------------------------------------------\n'
        return model_summary

    def summary(self):
        print(str(self))

    def fit(self, x_train: np.ndarray,
            epochs: int,
            batch_size: int | None = None,
            verbose: bool = True,
            metrics: list | None = None,
            random_state: int | None = None,
            validation_data: tuple | None = None,
            validation_split: float | None = None,
            callbacks: list = []) -> dict:
        """
        Fit the autoencoder to the training data.

        Args:
            x_train: Training data (input and target are the same for autoencoders)
            epochs: Number of training epochs
            batch_size: Number of samples per gradient update
            verbose: Whether to print training progress
            metrics: List of metrics to evaluate the model
            random_state: Random seed for shuffling
            validation_data: Tuple of validation data
            validation_split: Fraction of data to use for validation
            callbacks: List of callback objects

        Returns:
            Dictionary containing the training history
        """

        if hasattr(self, 'metrics') and self.metrics is not None:
            metrics = self.metrics

        seed = random_state if random_state is not None else self.random_state

        history = History({
            'loss': [],
            'val_loss': []
        })

        if validation_data is not None and validation_split is not None:
            raise ValueError("Cannot specify both validation_data and validation_split")
        elif validation_data is None and validation_split is not None:
            x_train, x_val = train_test_split(
                x_train,
                test_size=validation_split,
                random_state=seed
            )
            validation_data = (x_val, x_val)

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train

        _assign_missing_seeds(self.encoder_layers, seed)
        _assign_missing_seeds(self.decoder_layers, seed, 1000)

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU, Unidirectional))
                            for layer in self.encoder_layers + self.decoder_layers)
        has_embedding = any(isinstance(layer, Embedding)
                        for layer in self.encoder_layers + self.decoder_layers)

        if has_lstm_or_gru and not has_embedding:
            if len(x_train.shape) != 3:
                raise ValueError(
                    "Input data must be 3D (batch_size, time_steps, features) for LSTM/GRU layers without Embedding"
                )
        elif has_embedding:
            if len(x_train.shape) != 2:
                raise ValueError(
                    "Input data must be 2D (batch_size, sequence_length) when using Embedding layer"
                )

        if validation_data is not None:
            x_val, y_val = validation_data if len(validation_data) == 2 else (
                validation_data[0], validation_data[0])
            x_val = np.array(x_val)
            y_val = np.array(y_val)

        if metrics is not None:
            metrics = [Metric(m) for m in metrics]
            for metric in metrics:
                history[metric.name] = []
                history[f'val_{metric.name}'] = []

        for layer in self.encoder_layers + self.decoder_layers:
            if isinstance(layer, TextVectorization):
                layer.adapt(x_train)
                break

        callbacks = callbacks if callbacks is not None else []

        logs = {
            'model': self,
            'params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': [m.name for m in (metrics or [])],
                'validation': validation_data is not None,
            }
        }

        for callback in callbacks:
            callback.on_train_begin(logs)

        # a single generator, so that the data is shuffled differently at each epoch (and reproducibly)
        shuffle_rng = np.random.default_rng(seed)

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()

                x_train_shuffled = x_train[shuffle_rng.permutation(x_train.shape[0])]

                error = 0
                predictions_list = []
                inputs_list = []

                if batch_size is not None:
                    num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)

                    for j in range(0, x_train.shape[0], batch_size):
                        batch_index = j // batch_size

                        x_batch = x_train_shuffled[j:j + batch_size]

                        batch_logs = {
                            'batch': batch_index,
                            'size': len(x_batch),
                            'model': self
                        }
                        for callback in callbacks:
                            callback.on_batch_begin(batch_index, batch_logs)

                        batch_error = self.train_on_batch(x_batch)
                        error += batch_error
                        # kept only for the metrics: the predictions of a whole epoch can be very large
                        if metrics is not None:
                            predictions_list.append(self.predictions)
                            inputs_list.append(x_batch)

                        batch_logs.update({
                            'loss': batch_error,
                        })

                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(predictions_list[-1], inputs_list[-1])
                                batch_metrics[metric.name] = batch_metric_value
                            batch_logs.update(batch_metrics)

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        if verbose:
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metric_value = metric(
                                        _concatenate(predictions_list),
                                        _concatenate(inputs_list)
                                    )
                                    metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                            progress_bar(
                                batch_index + 1,
                                num_batches,
                                message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error / (batch_index + 1))} - {metrics_str}{time.time() - start_time:.2f}s'
                            )

                    error /= num_batches

                else:
                    error = self.train_on_batch(x_train_shuffled)
                    # kept only for the metrics: the predictions of a whole epoch can be very large
                    if metrics is not None:
                        predictions_list.append(self.predictions)
                        inputs_list.append(x_train_shuffled)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(
                                    _concatenate(predictions_list),
                                    _concatenate(inputs_list)
                                )
                                metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                        progress_bar(
                            1, 1,
                            message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error)} - {metrics_str}{time.time() - start_time:.2f}s'
                        )

                history['loss'].append(error)

                epoch_logs.update({
                    'loss': error,
                    'time': time.time() - start_time
                })

                if metrics is not None:
                    for metric in metrics:
                        metric_value = metric(
                            _concatenate(predictions_list),
                            _concatenate(inputs_list)
                        )
                        history[metric.name].append(metric_value)
                        epoch_logs[metric.name] = metric_value

                if validation_data is not None:
                    val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

                    if verbose:
                        print(f' - val_loss: {format_number(val_loss)}', end='')

                    if metrics is not None:
                        val_metrics = []
                        for metric in metrics:
                            val_metric = metric(val_predictions, y_val)
                            history[f'val_{metric.name}'].append(val_metric)
                            epoch_logs[f'val_{metric.name}'] = val_metric
                            val_metrics.append(val_metric)

                        if verbose:
                            val_metrics_str = ' - '.join(
                                f'val_{metric.name}: {format_number(val_metric)}'
                                for metric, val_metric in zip(metrics, val_metrics)
                            )
                            print(f' - {val_metrics_str}', end='')

                    val_predictions = None

                stop_training = False
                for callback in callbacks:
                    if callback.on_epoch_end(epoch, epoch_logs):
                        stop_training = True
                        break

                if verbose:
                    print()

                if stop_training:
                    break

        finally:
            final_logs = {
                'model': self,
                'history': history
            }
            for callback in callbacks:
                callback.on_train_end(final_logs)

            if verbose:
                print()

        return history

    def generate_image(self, x_train: np.ndarray, n_samples: int = 10, seed: int | None = None, n_examples: int = 1000) -> np.ndarray:
        if not self.variational:
            raise ValueError("generate_image requires variational=True")

        # statistics of the latent distribution on real examples
        self.forward_pass(x_train[:n_examples], training=False)

        mu = np.mean(self.latent_mean, axis=0)
        sigma = np.exp(0.5 * np.mean(self.latent_log_var, axis=0))

        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        noise = rng.standard_normal(size=(n_samples, mu.shape[0]))
        z = mu[None, :] + noise * sigma[None, :]

        generated = z
        for layer in self.decoder_layers:
            generated = _layer_forward(layer, generated, False)

        return generated


class Transformer(BaseModel):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 dropout_rate: float = 0.1,
                 max_sequence_length: int = 512,
                 gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = True,
                 padding_size: int = 32,
                 scale_embeddings: bool = True,
                 random_state: int | None = None,
                 ) -> None:

        super().__init__(gradient_clip_threshold,
                         enable_padding, padding_size, random_state)

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.scale_embeddings = scale_embeddings
        self.gradient_norms = {}

        # when the embeddings are scaled by sqrt(d_model), they must be initialized with a 1/sqrt(d_model) standard
        # deviation, otherwise they are much larger than the positional encoding (which would be ignored)
        embedding_init = "normal" if scale_embeddings else "default"

        # each sub-layer gets its own seed, otherwise all the layers would be initialized identically
        self.src_embedding = Embedding(
            self.src_vocab_size, d_model, input_length=max_sequence_length, weights_init=embedding_init,
            random_state=_derive_seed(random_state, 0))
        self.tgt_embedding = Embedding(
            self.tgt_vocab_size, d_model, input_length=max_sequence_length, weights_init=embedding_init,
            random_state=_derive_seed(random_state, 1))

        self.positional_encoding = PositionalEncoding(
            max_sequence_length=max_sequence_length,
            embedding_dim=d_model,
            scale_embeddings=scale_embeddings
        )

        self.encoder_dropout = Dropout(dropout_rate, random_state=_derive_seed(random_state, 2))
        self.encoder_layers: list = []
        for i in range(n_encoder_layers):
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                attention_dropout=dropout_rate,
                random_state=_derive_seed(random_state, 100 + i),
            )
            self.encoder_layers.append(encoder_layer)

        self.decoder_dropout = Dropout(dropout_rate, random_state=_derive_seed(random_state, 3))
        self.decoder_layers: list = []
        for i in range(n_decoder_layers):
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                attention_dropout=dropout_rate,
                random_state=_derive_seed(random_state, 200 + i)
            )
            self.decoder_layers.append(decoder_layer)

        self.output_layer = Dense(tgt_vocab_size, random_state=_derive_seed(random_state, 4))

        self.optimizer = None
        self.loss_function = None

    def _layers_with_keys(self) -> list:
        layers = [('src_embedding', self.src_embedding),
                  ('tgt_embedding', self.tgt_embedding),
                  ('positional_encoding', self.positional_encoding)]
        layers += [(f'encoder.{i}', layer) for i, layer in enumerate(self.encoder_layers)]
        layers += [(f'decoder.{i}', layer) for i, layer in enumerate(self.decoder_layers)]
        layers.append(('output_layer', self.output_layer))
        return layers

    def _all_layers(self) -> list:
        return [layer for _, layer in self._layers_with_keys()]

    def create_padding_mask(self, seq: np.ndarray) -> np.ndarray:
        if len(seq.shape) == 1:
            seq = seq[np.newaxis, :]
        mask = (seq == self.PAD_IDX).astype(np.bool_)
        return mask[:, np.newaxis, np.newaxis, :]

    def create_look_ahead_mask(self, size: int) -> np.ndarray:
        mask = np.triu(np.ones((size, size)), k=1).astype(np.bool_)
        return mask[np.newaxis, np.newaxis, :, :]

    def create_masks(self, inp: np.ndarray, tar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = inp.shape[0]
        enc_padding_mask = self.create_padding_mask(inp)
        # the keys of the cross-attention are the encoder outputs: the padding of the source sequence is masked
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tar.shape[1])
        dec_target_padding_mask = self.create_padding_mask(tar)

        look_ahead_mask = np.broadcast_to(
            look_ahead_mask,
            (batch_size, 1, tar.shape[1], tar.shape[1])
        )
        combined_mask = np.logical_or(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def encode(self, inp: np.ndarray, training: bool = True, mask: np.ndarray | None = None) -> np.ndarray:
        x = self.src_embedding.forward_pass(inp)
        x = self.positional_encoding.forward_pass(x, training=training)
        x = self.encoder_dropout.forward_pass(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward_pass(x, mask=mask, training=training)

        return x

    def decode(self, tar: np.ndarray, enc_output: np.ndarray, training: bool = True,
               look_ahead_mask: np.ndarray | None = None,
               padding_mask: np.ndarray | None = None) -> np.ndarray:

        x = self.tgt_embedding.forward_pass(tar)
        x = self.positional_encoding.forward_pass(x, training=training)
        x = self.decoder_dropout.forward_pass(x, training=training)

        attention_weights = []
        for decoder_layer in self.decoder_layers:
            x = decoder_layer.forward_pass(
                x, enc_output,
                self_attention_mask=look_ahead_mask,
                cross_attention_mask=padding_mask,
                training=training
            )
            attention_weights.append(
                decoder_layer.self_attention.attention_weights)

        self.last_attention_weights = attention_weights

        return x

    def forward_pass(self, inputs: tuple[np.ndarray, np.ndarray], training: bool = True) -> np.ndarray:
        """Returns the logits (the softmax is applied by the loss computation and by `predict`)."""
        encoder_input, decoder_input = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            encoder_input, decoder_input)

        enc_output = self.encode(encoder_input, training, enc_padding_mask)

        dec_output = self.decode(
            decoder_input,
            enc_output,
            training,
            look_ahead_mask,
            dec_padding_mask
        )

        output = self.output_layer.forward_pass(dec_output)

        return output

    def backward_pass(self, error: np.ndarray) -> None:
        error = self._clip(error)

        dx = self.output_layer.backward_pass(error)
        d_enc_output = None

        for decoder_layer in reversed(self.decoder_layers):
            dx, d_enc = decoder_layer.backward_pass(dx)
            d_enc_output = d_enc if d_enc_output is None else d_enc_output + d_enc

        dx = self.decoder_dropout.backward_pass(dx)
        # the positional encoding layer (shared by the encoder and the decoder) scales the embeddings
        dx = self.positional_encoding.backward_pass(dx)
        decoder_pe_gradient = self.positional_encoding.d_weights.copy() if self.positional_encoding.trainable else None
        self.tgt_embedding.backward_pass(dx)

        dx_enc = d_enc_output
        for encoder_layer in reversed(self.encoder_layers):
            dx_enc = encoder_layer.backward_pass(dx_enc)

        dx_enc = self.encoder_dropout.backward_pass(dx_enc)
        dx_enc = self.positional_encoding.backward_pass(dx_enc)
        if decoder_pe_gradient is not None:
            # gradients of both uses of the shared layer, counted as a single training step
            self.positional_encoding.d_weights = self.positional_encoding.d_weights + decoder_pe_gradient
            self.positional_encoding.current_step -= 1
        self.src_embedding.backward_pass(dx_enc)

    def prepare_data(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """Prepare data for text translation (we assume that the input and output sequences are already tokenized)"""
        if isinstance(x_train, np.ndarray):
            x_train = x_train.tolist()
        if isinstance(y_train, np.ndarray):
            y_train = y_train.tolist()

        if x_train[0][0] != self.SOS_IDX and x_train[0][-1] != self.EOS_IDX and y_train[0][0] != self.SOS_IDX and y_train[0][-1] != self.EOS_IDX:
            decoder_input = [[self.SOS_IDX] + seq for seq in y_train]
            decoder_target = [seq + [self.EOS_IDX] for seq in y_train]
        else:
            decoder_input = [seq[:-1] for seq in y_train]
            decoder_target = [seq[1:] for seq in y_train]

        encoder_input = pad_sequences(x_train,
                                      max_length=self.max_sequence_length,
                                      padding='post',
                                      pad_value=self.PAD_IDX)

        # the decoder sequences are only padded to the longest one: padding them to max_sequence_length (needed
        # for the source sequences) would multiply the cost of the decoder and of the output layer for nothing
        decoder_length = min(self.max_sequence_length, max(len(seq) for seq in decoder_input))

        decoder_input = pad_sequences(decoder_input,
                                      max_length=decoder_length,
                                      padding='post',
                                      pad_value=self.PAD_IDX)

        decoder_target = pad_sequences(decoder_target,
                                       max_length=decoder_length,
                                       padding='post',
                                       pad_value=self.PAD_IDX)

        return encoder_input, decoder_input, decoder_target

    def compile(self,
                loss_function: LossFunction | str,
                optimizer: Optimizer | str,
                verbose: bool = False,
                metrics: list | None = None) -> None:

        self.loss_function = loss_function if isinstance(
            loss_function, LossFunction) else LossFunction.from_name(loss_function)
        self.optimizer = optimizer if isinstance(
            optimizer, Optimizer) else Optimizer.from_name(optimizer)

        self.metrics = metrics

        if verbose:
            print(str(self))

    def update_weights(self) -> None:
        for key, layer in self._layers_with_keys():
            _update_layer_parameters(layer, key, self.optimizer)

    def _output_gradient(self, target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Gradient of the loss w.r.t. the logits (the loss being computed on the softmax of the logits)."""
        if isinstance(self.loss_function, CrossEntropyWithLabelSmoothing):
            labels = np.asarray(target).astype(int)
            mask = (labels != self.loss_function.ignore_index)
            valid_tokens = np.sum(mask)
            if valid_tokens == 0:
                return np.zeros_like(probabilities)
            one_hot = np.zeros_like(probabilities)
            np.put_along_axis(one_hot, labels[..., np.newaxis], 1.0, axis=-1)
            smoothing = self.loss_function.label_smoothing
            smooth_one_hot = (1.0 - smoothing) * one_hot + smoothing / probabilities.shape[-1]
            return (probabilities - smooth_one_hot) * mask[..., np.newaxis] / valid_tokens

        gradient = self.loss_function.derivative(target, probabilities)
        return probabilities * (gradient - np.sum(gradient * probabilities, axis=-1, keepdims=True))

    def train_on_batch(self, x_batch: tuple[np.ndarray, np.ndarray], y_batch: np.ndarray) -> float:
        self.gradient_norms = {}

        if isinstance(x_batch, (list, tuple)) and len(x_batch) == 2:
            encoder_input, decoder_input = x_batch
        else:
            raise ValueError("x_batch must be a list of [encoder_input, decoder_input]")

        decoder_target = y_batch

        self.predictions = self.forward_pass((encoder_input, decoder_input), training=True)
        # the output layer gives logits, the losses work on probabilities
        probabilities = softmax(self.predictions)

        loss = float(self.loss_function(decoder_target, probabilities))
        error = self._output_gradient(decoder_target, probabilities)

        self.backward_pass(error)

        self.update_weights()

        return loss

    def fit(self, x_train: np.ndarray | list, y_train: np.ndarray | list,
            epochs: int,
            batch_size: int | None = None,
            verbose: bool = True,
            metrics: list | None = None,
            random_state: int | None = None,
            validation_data: tuple | None = None,
            validation_split: float | None = None,
            callbacks: list = []) -> dict:

        if hasattr(self, 'metrics') and self.metrics is not None:
            metrics = self.metrics

        seed = random_state if random_state is not None else self.random_state

        history = History({
            'loss': [],
            'val_loss': []
        })

        if validation_data is not None and validation_split is not None:
            raise ValueError("Cannot specify both validation_data and validation_split")
        elif validation_data is None and validation_split is not None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=validation_split, random_state=seed)
            validation_data = (x_val, y_val)

        encoder_input, decoder_input, decoder_target = self.prepare_data(x_train, y_train)

        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)) and len(validation_data) == 2:
                x_val, y_val = validation_data
                x_val_enc, x_val_dec, y_val_prep = self.prepare_data(x_val, y_val)
            else:
                raise ValueError("validation_data must be a tuple of (x_val, y_val)")

        if metrics is not None:
            metrics = [Metric(m) for m in metrics]
            for metric in metrics:
                history[metric.name] = []
                history[f'val_{metric.name}'] = []

        callbacks = callbacks if callbacks is not None else []

        logs = {
            'model': self,
            'params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': [m.name for m in (metrics or [])],
                'validation': validation_data is not None,
            }
        }

        for callback in callbacks:
            callback.on_train_begin(logs)

        # a single generator, so that the data is shuffled differently at each epoch (and reproducibly)
        shuffle_rng = np.random.default_rng(seed)

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()

                indices = shuffle_rng.permutation(len(encoder_input))
                encoder_input_shuffled = encoder_input[indices]
                decoder_input_shuffled = decoder_input[indices]
                decoder_target_shuffled = decoder_target[indices]

                error = 0
                predictions_list = []
                y_true_list = []

                if batch_size is not None:
                    num_batches = np.ceil(len(encoder_input) / batch_size).astype(int)

                    for j in range(0, len(encoder_input), batch_size):
                        batch_index = j // batch_size

                        enc_batch = encoder_input_shuffled[j:j + batch_size]
                        dec_batch = decoder_input_shuffled[j:j + batch_size]
                        target_batch = decoder_target_shuffled[j:j + batch_size]

                        batch_logs = {
                            'batch': batch_index,
                            'size': len(enc_batch),
                            'model': self
                        }

                        for callback in callbacks:
                            callback.on_batch_begin(batch_index, batch_logs)

                        batch_error = self.train_on_batch([enc_batch, dec_batch], target_batch)
                        error += batch_error
                        # kept only for the metrics: the predictions of a whole epoch can be very large
                        if metrics is not None:
                            predictions_list.append(self.predictions)
                            y_true_list.append(target_batch)

                        batch_logs.update({'loss': batch_error})

                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(predictions_list[-1], y_true_list[-1])
                                batch_metrics[metric.name] = batch_metric_value
                            batch_logs.update(batch_metrics)

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        if verbose:
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metric_value = metric(_concatenate(predictions_list),
                                                        _concatenate(y_true_list))
                                    metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                            progress_bar(batch_index + 1, num_batches,
                                    message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error / (batch_index + 1))} - {metrics_str}{time.time() - start_time:.2f}s')

                    error /= num_batches
                else:
                    error = self.train_on_batch([encoder_input_shuffled, decoder_input_shuffled], decoder_target_shuffled)
                    # kept only for the metrics: the predictions of a whole epoch can be very large
                    if metrics is not None:
                        predictions_list.append(self.predictions)
                        y_true_list.append(decoder_target_shuffled)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(_concatenate(predictions_list),
                                                _concatenate(y_true_list))
                                metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                        progress_bar(1, 1,
                                message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error)} - {metrics_str}{time.time() - start_time:.2f}s')

                history['loss'].append(error)

                epoch_logs.update({
                    'loss': error,
                    'time': time.time() - start_time
                })

                if metrics is not None:
                    for metric in metrics:
                        metric_value = metric(_concatenate(predictions_list),
                                        _concatenate(y_true_list))
                        history[metric.name].append(metric_value)
                        epoch_logs[metric.name] = metric_value

                if validation_data is not None:
                    val_loss, val_predictions = self.evaluate([x_val_enc, x_val_dec],
                                                            y_val_prep, batch_size)

                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

                    if verbose:
                        print(f' - val_loss: {format_number(val_loss)}', end='')

                    if metrics is not None:
                        val_metrics = []
                        for metric in metrics:
                            # the predictions are compared to the prepared (shifted and padded) targets
                            val_metric = metric(val_predictions, y_val_prep)
                            history[f'val_{metric.name}'].append(val_metric)
                            epoch_logs[f'val_{metric.name}'] = val_metric
                            val_metrics.append(val_metric)

                        if verbose:
                            val_metrics_str = ' - '.join(
                                f'val_{metric.name}: {format_number(val_metric)}'
                                for metric, val_metric in zip(metrics, val_metrics)
                            )
                            print(f' - {val_metrics_str}', end='')

                    val_predictions = None

                stop_training = False
                for callback in callbacks:
                    if callback.on_epoch_end(epoch, epoch_logs):
                        stop_training = True
                        break

                if verbose:
                    print()

                if stop_training:
                    break

        finally:
            final_logs = {
                'model': self,
                'history': history
            }
            for callback in callbacks:
                callback.on_train_end(final_logs)

            if verbose:
                print()

        return history

    def predict(self, inp: np.ndarray, max_length: int = 50, beam_size: int = 5,
                alpha: float = 0.6, min_length: int = 3, temperature: float = 0.5) -> np.ndarray:
        inp = np.asarray(inp)
        if inp.ndim == 1:
            inp = inp[np.newaxis, :]

        # same masking of the source padding as during training
        enc_padding_mask = self.create_padding_mask(inp)
        enc_output = self.encode(inp, training=False, mask=enc_padding_mask)
        input_length = np.sum(inp[0] != self.PAD_IDX)
        adaptive_max_length = min(max_length, max(input_length * 2, 10))

        def ranking_score(seq: np.ndarray, log_prob: float) -> float:
            # length normalized log probability (GNMT), with a penalty for sequences much longer than the input
            length = seq.shape[1]
            length_penalty = ((5 + length) / 6) ** alpha
            too_long_penalty = max(0.0, length - input_length * 1.5) * 0.2
            return log_prob / length_penalty - too_long_penalty

        # each beam holds a sequence and its (raw) cumulative log probability
        beams = [(np.array([[self.SOS_IDX]]), 0.0)]

        for _ in range(adaptive_max_length - 1):
            all_candidates = []

            for seq, log_prob in beams:

                if seq[0, -1] == self.EOS_IDX:
                    all_candidates.append((seq, log_prob))
                    continue

                dec_output = self.decode(
                    seq,
                    enc_output,
                    training=False,
                    look_ahead_mask=self.create_look_ahead_mask(seq.shape[1]),
                    padding_mask=enc_padding_mask
                )

                logits = self.output_layer.forward_pass(dec_output)[:, -1, :].copy()

                invalid_tokens = [self.PAD_IDX, self.SOS_IDX, self.UNK_IDX]

                for token in invalid_tokens:
                    logits[:, token] = -np.inf

                if seq.shape[1] < min_length:
                    logits[:, self.EOS_IDX] = -np.inf

                log_probs = log_softmax(logits[0] / temperature)

                top_k = min(beam_size * 2, self.tgt_vocab_size)
                top_indices = np.argpartition(log_probs, -top_k)[-top_k:]

                for idx in top_indices:
                    if not np.isfinite(log_probs[idx]):
                        continue
                    candidate_seq = np.concatenate([seq, [[idx]]], axis=1)
                    all_candidates.append((candidate_seq, log_prob + log_probs[idx]))

            if not all_candidates:
                break

            beams = sorted(all_candidates, key=lambda c: ranking_score(*c), reverse=True)[:beam_size]

            if all(seq[0, -1] == self.EOS_IDX for seq, _ in beams):
                break

        if not beams:
            return np.array([[]])

        best_seq = max(beams, key=lambda c: ranking_score(*c))[0]

        result = best_seq[:, 1:]
        if result.shape[1] > 0 and result[0, -1] == self.EOS_IDX:
            result = result[:, :-1]

        return result

    def evaluate(self, x_test: list[np.ndarray], y_test: np.ndarray, batch_size: int = 32) -> tuple[float, np.ndarray]:
        if isinstance(x_test, (list, tuple)) and len(x_test) == 2:
            encoder_input, decoder_input = x_test
        else:
            raise ValueError(
                "x_test must be a list of [encoder_input, decoder_input]")

        decoder_target = y_test

        total_loss = 0
        if batch_size is None:
            batch_size = len(encoder_input)
        predictions_list = []

        for i in range(0, len(encoder_input), batch_size):
            enc_batch = encoder_input[i:i + batch_size]
            dec_batch = decoder_input[i:i + batch_size]
            target_batch = decoder_target[i:i + batch_size]

            predictions = self.forward_pass(
                (enc_batch, dec_batch), training=False)
            batch_loss = self.loss_function(target_batch, softmax(predictions))

            total_loss += batch_loss * len(enc_batch)
            predictions_list.append(predictions)

        avg_loss = float(total_loss / len(encoder_input))
        all_predictions = _concatenate(predictions_list)

        return avg_loss, all_predictions

    def get_config(self) -> dict:
        config = {
            'type': 'Transformer',
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'max_sequence_length': self.max_sequence_length,
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'scale_embeddings': self.scale_embeddings,
            'random_state': self.random_state,

            'src_embedding': self.src_embedding.get_config(),
            'tgt_embedding': self.tgt_embedding.get_config(),
            'positional_encoding': self.positional_encoding.get_config(),

            'encoder_layers': [layer.get_config() for layer in self.encoder_layers],
            'decoder_layers': [layer.get_config() for layer in self.decoder_layers],

            'encoder_dropout': self.encoder_dropout.get_config(),
            'decoder_dropout': self.decoder_dropout.get_config(),

            'output_layer': self.output_layer.get_config(),

            'loss_function': self.loss_function.get_config() if self.loss_function is not None else None,
            'optimizer': self.optimizer.get_config() if self.optimizer is not None else None
        }
        return config

    @classmethod
    def load(cls, filename: str) -> 'Transformer':
        with open(filename, 'r') as f:
            config = json.load(f)

        if config['type'] != 'Transformer':
            raise ValueError(f"Invalid model type {config['type']}")

        model = cls(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            d_ff=config['d_ff'],
            dropout_rate=config['dropout_rate'],
            max_sequence_length=config['max_sequence_length'],
            gradient_clip_threshold=config['gradient_clip_threshold'],
            enable_padding=config['enable_padding'],
            padding_size=config['padding_size'],
            scale_embeddings=config.get('scale_embeddings', True),
            random_state=config['random_state']
        )

        model.src_embedding = Embedding.from_config(config['src_embedding'])
        model.tgt_embedding = Embedding.from_config(config['tgt_embedding'])
        model.positional_encoding = PositionalEncoding.from_config(config['positional_encoding'])

        model.encoder_dropout = Dropout.from_config(config['encoder_dropout'])
        model.decoder_dropout = Dropout.from_config(config['decoder_dropout'])

        model.encoder_layers = [TransformerEncoderLayer.from_config(layer_config)
                            for layer_config in config['encoder_layers']]
        model.decoder_layers = [TransformerDecoderLayer.from_config(layer_config)
                            for layer_config in config['decoder_layers']]

        model.output_layer = Dense.from_config(config['output_layer'])

        if config['loss_function']:
            model.loss_function = LossFunction.from_config(config['loss_function'])
        if config['optimizer']:
            model.optimizer = Optimizer.from_config(config['optimizer'])

        return model

    def save(self, filename: str) -> None:
        base, ext = os.path.splitext(filename)

        config = self.get_config()

        if self.src_embedding is not None:
            src_emb_file = f"{base}_src_embedding{ext}"
            config['src_embedding_file'] = src_emb_file
            with open(src_emb_file, 'w') as f:
                json.dump(self.src_embedding.get_config(), f, indent=4, default=to_json_serializable)

        if self.tgt_embedding is not None:
            tgt_emb_file = f"{base}_tgt_embedding{ext}"
            config['tgt_embedding_file'] = tgt_emb_file
            with open(tgt_emb_file, 'w') as f:
                json.dump(self.tgt_embedding.get_config(), f, indent=4, default=to_json_serializable)

        config['encoder_layers_files'] = []
        for i, layer in enumerate(self.encoder_layers):
            encoder_file = f"{base}_encoder_layer_{i}{ext}"
            config['encoder_layers_files'].append(encoder_file)
            with open(encoder_file, 'w') as f:
                json.dump(layer.get_config(), f, indent=4, default=to_json_serializable)

        config['decoder_layers_files'] = []
        for i, layer in enumerate(self.decoder_layers):
            decoder_file = f"{base}_decoder_layer_{i}{ext}"
            config['decoder_layers_files'].append(decoder_file)
            with open(decoder_file, 'w') as f:
                json.dump(layer.get_config(), f, indent=4, default=to_json_serializable)

        if self.output_layer is not None:
            output_file = f"{base}_output_layer{ext}"
            config['output_layer_file'] = output_file
            with open(output_file, 'w') as f:
                json.dump(self.output_layer.get_config(), f, indent=4, default=to_json_serializable)

        if self.optimizer is not None:
            optimizer_file = f"{base}_optimizer{ext}"
            config['optimizer_file'] = optimizer_file
            with open(optimizer_file, 'w') as f:
                json.dump(self.optimizer.get_config(), f, indent=4, default=to_json_serializable)

        if self.loss_function is not None:
            loss_file = f"{base}_loss{ext}"
            config['loss_file'] = loss_file
            with open(loss_file, 'w') as f:
                json.dump(self.loss_function.get_config(), f, indent=4, default=to_json_serializable)

        with open(filename, 'w') as f:
            json.dump(config, f, indent=4, default=to_json_serializable)

    def __str__(self) -> str:
        return (f"Transformer(\n"
                f"  src_vocab_size={self.src_vocab_size},\n"
                f"  tgt_vocab_size={self.tgt_vocab_size},\n"
                f"  d_model={self.d_model},\n"
                f"  n_heads={self.n_heads},\n"
                f"  n_encoder_layers={self.n_encoder_layers},\n"
                f"  n_decoder_layers={self.n_decoder_layers},\n"
                f"  d_ff={self.d_ff},\n"
                f"  dropout_rate={self.dropout_rate},\n"
                f"  max_sequence_length={self.max_sequence_length}\n"
                f")")


class GAN(BaseModel):
    def __init__(
        self,
        latent_dim: int = 100,
        n_classes: int | None = None,
        gradient_clip_threshold: float = 0.1,
        enable_padding: bool = False,
        padding_size: int = 32,
        random_state: int | None = None,
        use_spectral_norm: bool = True,
        use_gradient_penalty: bool = True,
        gp_weight: float = 10.0,
        image_height: int | None = None,
        image_width: int | None = None,
        label_smoothing: float = 0.9
    ):
        super().__init__(gradient_clip_threshold, enable_padding, padding_size, random_state)

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.generator_loss = None
        self.discriminator_loss = None
        self.use_spectral_norm = use_spectral_norm
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_weight = gp_weight
        self._image_height = image_height
        self._image_width = image_width
        self.label_smoothing = label_smoothing

        self.spectral_norm = SpectralNorm()
        self._train_step = 0
        self.gradient_debugger = GradientDebugger(
            clip_threshold=gradient_clip_threshold)

    def _get_rng(self) -> np.random.Generator:
        # a single generator: the latent points and the real batches must change at every training step
        if getattr(self, '_rng', None) is None:
            self._rng = np.random.default_rng(self.random_state)
        return self._rng

    def _all_layers(self) -> list:
        layers = []
        if self.generator is not None:
            layers += list(self.generator.layers)
        if self.discriminator is not None:
            layers += list(self.discriminator.layers)
        return layers

    @property
    def last_activation(self) -> str | None:
        """Name of the activation function of the generator's output (used to rescale the plotted images)."""
        if self.generator is None or not self.generator.layers:
            return None
        last_layer = self.generator.layers[-1]
        if isinstance(last_layer, Activation):
            return last_layer.activation_function.get_config()['name'].lower()
        return None

    @property
    def image_dimensions(self) -> tuple[int, int]:
        if self._image_height is None or self._image_width is None:
            if self.generator is None:
                raise ValueError(
                    "The image dimensions are not defined and the generator is not compiled.")
            return self._infer_dimensions_from_generator()
        return self._image_height, self._image_width

    def _generator_output_shape(self) -> tuple:
        """Shape of a generated sample (without the batch axis)."""
        noise_size = self.latent_dim + (self.n_classes if self.n_classes is not None else 0)
        return self.generator.forward_pass(np.zeros((1, noise_size)), training=False).shape[1:]

    def _infer_dimensions_from_generator(self) -> tuple[int, int]:
        output_shape = self._generator_output_shape()

        # images generated with their spatial dimensions, e.g. (height, width, channels)
        if len(output_shape) >= 2:
            return int(output_shape[0]), int(output_shape[1])

        # flattened images: the most square factorization of the number of pixels
        n_pixels = int(output_shape[0])

        height = int(np.sqrt(n_pixels))
        while n_pixels % height != 0:
            height -= 1
        width = n_pixels // height

        return height, width

    def compile(
        self,
        generator: 'Sequential',
        discriminator: 'Sequential',
        generator_optimizer: Optimizer | str,
        discriminator_optimizer: Optimizer | str,
        loss_function: LossFunction | str = 'bce',
        verbose: bool = False,
        metrics: list | None = None
    ):
        if self.n_classes is not None:
            generator.n_classes = self.n_classes
            discriminator.n_classes = self.n_classes

        self.generator = generator
        self.discriminator = discriminator

        # like in the other models, the layers without explicit seed get one derived from the seed of the GAN
        # (the weights of the generator are initialized below, when its output shape is computed)
        _assign_missing_seeds(generator.layers, _derive_seed(self.random_state, 0))
        _assign_missing_seeds(discriminator.layers, _derive_seed(self.random_state, 1))

        if self.n_classes is not None:
            noise_size = self.latent_dim + self.n_classes
            if generator.layers[0].input_dim != noise_size:
                raise ValueError(
                    f"Generator input dimension ({generator.layers[0].input_dim}) "
                    f"does not match expected size (latent_dim + n_classes = {noise_size})"
                )

        if self._image_height is None or self._image_width is None:
            self._image_height, self._image_width = self._infer_dimensions_from_generator()
            if verbose:
                print(
                    f"Inferred image dimensions: {self._image_height}x{self._image_width}")

        # the generated samples (and not the last Dense layer, which may be followed by convolutions) must be images
        # of the expected size
        output_shape = self._generator_output_shape()
        generated_size = int(np.prod(output_shape[:2])) if len(output_shape) >= 2 else int(output_shape[0])
        expected_size = self._image_height * self._image_width
        if generated_size != expected_size:
            raise ValueError(
                f"The generator must produce images of size {expected_size} "
                f"({self._image_height}x{self._image_width}), "
                f"but it produces samples of shape {output_shape}."
            )

        self.generator_optimizer = (
            generator_optimizer if isinstance(generator_optimizer, Optimizer)
            else Optimizer.from_name(generator_optimizer)
        )
        self.discriminator_optimizer = (
            discriminator_optimizer if isinstance(discriminator_optimizer, Optimizer)
            else Optimizer.from_name(discriminator_optimizer)
        )

        self.generator_loss = (
            loss_function if isinstance(loss_function, LossFunction)
            else LossFunction.from_name(loss_function)
        )
        self.discriminator_loss = (
            loss_function if isinstance(loss_function, LossFunction)
            else LossFunction.from_name(loss_function)
        )

        self.generator.loss_function = self.generator_loss
        self.generator.optimizer = self.generator_optimizer
        self.discriminator.loss_function = self.discriminator_loss
        self.discriminator.optimizer = self.discriminator_optimizer

        self.metrics = metrics

        if verbose:
            print(str(self))

    def forward_pass(self, latent_vectors: np.ndarray, training: bool = True) -> np.ndarray:
        if self.generator is None:
            raise ValueError("Model must be compiled before forward pass")

        return self.generator.forward_pass(latent_vectors, training)

    def backward_pass(self, error: np.ndarray):
        if self.generator is None:
            raise ValueError("Model must be compiled before backward pass")

        self.generator.backward_pass(error)

    def _generate_latent_points(self, n_samples: int, labels: np.ndarray | None = None,
                                rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng if rng is not None else self._get_rng()

        if self.n_classes is not None:
            if labels is not None:
                labels = np.asarray(labels)
                if labels.ndim == 1:
                    one_hot_labels = np.zeros((len(labels), self.n_classes))
                    one_hot_labels[np.arange(len(labels)), labels.astype(int)] = 1
                    labels = one_hot_labels
                elif labels.shape[1] != self.n_classes:
                    raise ValueError(f"Labels must have {self.n_classes} columns when one-hot encoded")

                latent_points = rng.normal(0, 1, (len(labels), self.latent_dim))
                latent_points = np.concatenate([latent_points, labels], axis=1)
                return latent_points, labels

            samples_per_class = n_samples // self.n_classes
            remaining_samples = n_samples % self.n_classes

            all_latent_points = []
            all_labels = []

            for class_idx in range(self.n_classes):
                n_samples_this_class = samples_per_class
                if class_idx < remaining_samples:
                    n_samples_this_class += 1

                class_noise = rng.normal(0, 1, (n_samples_this_class, self.latent_dim))

                class_labels = np.zeros((n_samples_this_class, self.n_classes))
                class_labels[:, class_idx] = 1

                class_latent = np.concatenate([class_noise, class_labels], axis=1)

                all_latent_points.append(class_latent)
                all_labels.append(class_labels)

            latent_points = np.concatenate(all_latent_points, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            return latent_points, labels
        else:
            latent_points = rng.normal(0, 1, (n_samples, self.latent_dim))
            return latent_points, None

    def _apply_spectral_norm(self, model: 'Sequential'):
        if not self.use_spectral_norm:
            return

        for layer in model.layers:
            if getattr(layer, 'weights', None) is not None:
                # in place, so that the optimizers keep working on the same arrays
                layer.weights[...] = self.spectral_norm(layer.weights)

    def _gradient_penalty(self, real_samples: np.ndarray, fake_samples: np.ndarray) -> float:
        if not self.use_gradient_penalty:
            return 0.0

        rng = self._get_rng()
        batch_size = real_samples.shape[0]
        alpha = rng.uniform(0, 1, (batch_size,) + (1,) * (real_samples.ndim - 1))

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        disc_interpolated = self.discriminator.forward_pass(interpolated)
        gradients = self.discriminator.backward_pass(
            np.ones_like(disc_interpolated),
            compute_only=True
        )

        gradients_norm = np.sqrt(np.sum(np.square(gradients.reshape(batch_size, -1)), axis=1))
        return self.gp_weight * np.mean(np.square(gradients_norm - 1.0))

    def _process_gradients(self, gradients: np.ndarray, name: str) -> np.ndarray:
        self.gradient_debugger.log_gradient_stats(
            name, gradients, self._train_step)
        return self.gradient_debugger.adaptive_clip_gradients(gradients)

    def _ensure_initialized(self, input_data: np.ndarray):
        if not hasattr(self.generator, '_initialized'):
            latent_points = self._generate_latent_points(1)
            self.generator.forward_pass(latent_points, training=False)
            self.generator._initialized = True

        if not hasattr(self.discriminator, '_initialized'):
            self.discriminator.forward_pass(input_data[:1], training=False)
            self.discriminator._initialized = True

    def train_on_batch(
        self,
        real_samples: np.ndarray,
        labels: np.ndarray | None = None,
        batch_size: int = 32,
        n_critic: int = 1
    ) -> tuple[float, float]:
        rng = self._get_rng()
        batch_size = min(batch_size, len(real_samples))

        d_loss_total = 0
        for _ in range(n_critic):
            idx = rng.choice(len(real_samples), batch_size, replace=False)
            real_batch = real_samples[idx]
            batch_labels = labels[idx] if labels is not None else None

            latent_points, gen_labels = self._generate_latent_points(batch_size, batch_labels)
            fake_batch = self.generator.forward_pass(latent_points, training=False)

            combined_batch = np.concatenate([real_batch, fake_batch])
            if self.n_classes is not None and batch_labels is not None:
                disc_labels = np.concatenate([gen_labels, gen_labels])
                discriminator_input = np.concatenate([combined_batch, disc_labels], axis=1)
            else:
                discriminator_input = combined_batch

            combined_labels = np.zeros((2 * batch_size, 1))
            combined_labels[:batch_size] = self.label_smoothing
            combined_labels[batch_size:] = 1 - self.label_smoothing

            self.discriminator.y_true = combined_labels
            predictions = self.discriminator.forward_pass(discriminator_input, training=True)
            d_loss = self.discriminator_loss(combined_labels, predictions)
            d_grad = self.discriminator_loss.derivative(combined_labels, predictions)
            self.discriminator.backward_pass(d_grad)

            d_loss_total += d_loss

        d_loss_avg = d_loss_total / n_critic

        latent_points, gen_labels = self._generate_latent_points(batch_size)
        fake_samples = self.generator.forward_pass(latent_points, training=True)

        if self.n_classes is not None:
            discriminator_input = np.concatenate([fake_samples, gen_labels], axis=1)
        else:
            discriminator_input = fake_samples

        target_labels = np.ones((batch_size, 1))

        disc_predictions = self.discriminator.forward_pass(discriminator_input, training=False)

        self.discriminator.y_true = target_labels
        g_loss = self.generator_loss(target_labels, disc_predictions)
        g_grad = self.generator_loss.derivative(target_labels, disc_predictions)

        d_grad = self.discriminator.backward_pass(g_grad, compute_only=True)
        if self.n_classes is not None:
            # the labels concatenated to the generated images are not generated: their gradient is dropped
            d_grad = d_grad[:, :fake_samples.shape[1]]
        self.generator.backward_pass(d_grad.reshape(fake_samples.shape), gan=True)

        self._train_step += 1

        return float(d_loss_avg), float(g_loss)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int | None = None,
        n_critic: int = 5,
        verbose: bool = True,
        metrics: list | None = None,
        random_state: int | None = None,
        validation_data: tuple | None = None,
        validation_split: float | None = None,
        callbacks: list = [],
        plot_generated: bool = False,
        plot_interval: int = 1,
        fixed_noise: np.ndarray | None = None,
        fixed_labels: np.ndarray | None = None,
        n_gen_samples: int | None = None,
        visualization_grid: tuple[int, int] = (8, 8)
    ) -> dict:

        if hasattr(self, 'metrics') and self.metrics is not None:
            metrics = self.metrics

        seed = random_state if random_state is not None else self.random_state

        history = History({
            'discriminator_loss': [],
            'generator_loss': [],
            'val_discriminator_loss': [],
            'val_generator_loss': []
        })

        if validation_data is not None and validation_split is not None:
            raise ValueError("Cannot specify both validation_data and validation_split")
        elif validation_data is None and validation_split is not None:
            if y_train is not None:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train,
                    test_size=validation_split,
                    random_state=seed
                )
                validation_data = (x_val, y_val)
            else:
                x_train, x_val = train_test_split(
                    x_train, test_size=validation_split, random_state=seed
                )
                validation_data = (x_val, None)

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
        # the labels are optional (non conditional GANs), all-zero labels are valid labels
        has_labels = y_train is not None
        if has_labels:
            y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        if metrics is not None:
            metrics = [Metric(m) for m in metrics]
            for metric in metrics:
                history[f'discriminator_{metric.name}'] = []
                history[f'generator_{metric.name}'] = []
                if validation_data is not None:
                    history[f'val_discriminator_{metric.name}'] = []
                    history[f'val_generator_{metric.name}'] = []

        # for a conditional GAN, each row of the visualization grid shows a class
        if self.n_classes is not None:
            visualization_grid = (self.n_classes, visualization_grid[1])

        if plot_generated:
            if fixed_noise is None:
                n_rows, n_cols = visualization_grid
                n_samples = n_rows * n_cols

                noise_rng = np.random.default_rng(seed)
                fixed_noise = noise_rng.normal(0, 1, (n_samples, self.latent_dim))

            if self.n_classes is not None and fixed_labels is None:
                fixed_labels = []
                for class_idx in range(self.n_classes):
                    class_labels = np.zeros((visualization_grid[1], self.n_classes))
                    class_labels[:, class_idx] = 1
                    fixed_labels.append(class_labels)
                fixed_labels = np.concatenate(fixed_labels, axis=0)

        callbacks = callbacks if callbacks is not None else []

        logs = {
            'model': self,
            'params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'n_critic': n_critic,
                'verbose': verbose,
                'metrics': [m.name for m in (metrics or [])],
                'validation': validation_data is not None,
                'plot_generated': plot_generated,
            }
        }

        for callback in callbacks:
            callback.on_train_begin(logs)

        # a single generator, so that the data is shuffled differently at each epoch (and reproducibly)
        shuffle_rng = np.random.default_rng(seed)
        # the noise of the metrics has its own generator: asking for metrics does not change the training
        metrics_rng = np.random.default_rng(_derive_seed(seed, 2))

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()
                permutation = shuffle_rng.permutation(x_train.shape[0])
                x_train_shuffled = x_train[permutation]
                y_train_shuffled = y_train[permutation] if has_labels else None

                d_error = 0
                g_error = 0

                metric_values = {
                    f'discriminator_{metric.name}': 0.0 for metric in (metrics or [])}
                metric_values.update(
                    {f'generator_{metric.name}': 0.0 for metric in (metrics or [])})

                if batch_size is not None:
                    num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)

                    for j in range(0, x_train.shape[0], batch_size):
                        batch_index = j // batch_size
                        x_batch = x_train_shuffled[j:j + batch_size]
                        y_batch = y_train_shuffled[j:j + batch_size] if has_labels else None

                        batch_logs = {
                            'batch': batch_index,
                            'size': len(x_batch),
                            'model': self
                        }

                        for callback in callbacks:
                            callback.on_batch_begin(batch_index, batch_logs)

                        d_loss, g_loss = self.train_on_batch(
                            x_batch, y_batch, min(batch_size, len(x_batch)), n_critic)
                        d_error += d_loss
                        g_error += g_loss

                        batch_metrics = {}
                        if metrics is not None:
                            latent_points, _ = self._generate_latent_points(len(x_batch), y_batch, rng=metrics_rng)
                            generated_samples = self.forward_pass(latent_points, training=False)
                            for metric in metrics:
                                metric_value = metric(generated_samples, x_batch)
                                metric_values[f'generator_{metric.name}'] += metric_value
                                metric_values[f'discriminator_{metric.name}'] += metric_value
                                batch_metrics[metric.name] = metric_value

                        batch_logs.update({
                            'discriminator_loss': d_loss,
                            'generator_loss': g_loss,
                            **batch_metrics
                        })

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        if verbose:
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metrics_str += f'{metric.name}: {format_number(batch_metrics[metric.name])} - '
                            progress_bar(
                                batch_index + 1,
                                num_batches,
                                message=(
                                    f'Epoch {epoch + 1}/{epochs} - '
                                    f'd_loss: {format_number(d_error / (batch_index + 1))} - '
                                    f'g_loss: {format_number(g_error / (batch_index + 1))} - '
                                    f'{metrics_str}'
                                    f'{time.time() - start_time:.2f}s'
                                )
                            )

                    d_error /= num_batches
                    g_error /= num_batches
                    for k in metric_values:
                        metric_values[k] /= num_batches

                else:
                    d_error, g_error = self.train_on_batch(x_train_shuffled, y_train_shuffled, len(x_train), n_critic)

                    if metrics is not None:
                        latent_points, _ = self._generate_latent_points(
                            len(x_train) if n_gen_samples is None else n_gen_samples,
                            y_train if n_gen_samples is None else None, rng=metrics_rng)
                        generated_samples = self.forward_pass(latent_points, training=False)

                        for metric in metrics:
                            metric_value = metric(generated_samples, x_train)
                            metric_values[f'generator_{metric.name}'] = metric_value
                            metric_values[f'discriminator_{metric.name}'] = metric_value

                    if verbose:
                        progress_bar(1, 1, message=(
                            f'Epoch {epoch + 1}/{epochs} - '
                            f'd_loss: {format_number(d_error)} - '
                            f'g_loss: {format_number(g_error)} - '
                            f'{time.time() - start_time:.2f}s'
                        ))

                history['discriminator_loss'].append(d_error)
                history['generator_loss'].append(g_error)

                for k, v in metric_values.items():
                    history[k].append(v)

                epoch_logs.update({
                    'discriminator_loss': d_error,
                    'generator_loss': g_error,
                    'time': time.time() - start_time,
                    **metric_values
                })

                if validation_data is not None:
                    if isinstance(validation_data, tuple):
                        x_val, y_val = validation_data if len(validation_data) == 2 else (validation_data[0], None)
                    else:
                        x_val = validation_data
                        y_val = None

                    x_val = np.array(x_val)

                    val_d_loss, val_g_loss = self.evaluate(x_val, y_val, batch_size if batch_size is not None else len(x_val))

                    history['val_discriminator_loss'].append(val_d_loss)
                    history['val_generator_loss'].append(val_g_loss)
                    epoch_logs.update({
                        'val_discriminator_loss': val_d_loss,
                        'val_generator_loss': val_g_loss
                    })

                    if verbose:
                        print(f' - val_d_loss: {format_number(val_d_loss)} '
                            f'- val_g_loss: {format_number(val_g_loss)}', end='')

                if plot_generated and (epoch + 1) % plot_interval == 0:
                    self._plot_samples(fixed_noise, epoch + 1, fixed_labels, visualization_grid)

                stop_training = False
                for callback in callbacks:
                    if callback.on_epoch_end(epoch, epoch_logs):
                        stop_training = True
                        break

                if verbose:
                    print()

                if stop_training:
                    break

        finally:
            final_logs = {
                'model': self,
                'history': history
            }
            for callback in callbacks:
                callback.on_train_end(final_logs)

            if verbose:
                print()

        return history

    def _plot_samples(self, noise: np.ndarray, epoch: int, labels: np.ndarray | None = None,
                    grid_size: tuple[int, int] = (8, 8)):
        import matplotlib.pyplot as plt

        n_rows, n_cols = grid_size
        n_samples = n_rows * n_cols

        if noise.shape[0] != n_samples:
            raise ValueError(f"The number of noise samples ({noise.shape[0]}) must match the grid size ({n_samples})")

        if self.n_classes is not None:
            if labels is None:
                labels = np.repeat(np.arange(self.n_classes), n_cols)
            labels = np.asarray(labels)
            if labels.ndim == 1:
                labels = np.eye(self.n_classes)[labels.astype(int)]
            # the fixed noise is used, so that the evolution of the same samples can be followed
            latent_points = np.concatenate([noise, labels], axis=1)

            generated = self.generator.forward_pass(latent_points, training=False)
        else:
            generated = self.generator.forward_pass(noise, training=False)

        if self.last_activation == 'tanh':
            generated = (generated + 1) * 0.5
        elif self.last_activation == 'sigmoid':
            pass
        else:
            generated = generated - generated.min()
            generated = generated / (generated.max() + 1e-8)

        height, width = self.image_dimensions
        figure = np.zeros((height * n_rows, width * n_cols))

        for i in range(n_rows):
            for j in range(n_cols):
                sample_idx = i * n_cols + j
                sample = generated[sample_idx].reshape(height, width)
                figure[i * height:(i + 1) * height, j * width:(j + 1) * width] = sample

        plt.figure(figsize=(10, 8))
        plt.imshow(figure, cmap='gray_r', interpolation='nearest')
        plt.axis('off')

        if self.n_classes is not None:
            for i in range(n_rows):
                plt.text(-width/2, i * height + height/2,
                        f'Class {i}',
                        horizontalalignment='right',
                        verticalalignment='center')

        plt.tight_layout(pad=0)
        plt.savefig(f'video{str(epoch).zfill(2)}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def predict(self, n_samples: int, labels: np.ndarray | None = None, temperature: float = 1.0) -> np.ndarray:
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 1:
                one_hot_labels = np.zeros((len(labels), self.n_classes))
                one_hot_labels[np.arange(len(labels)), labels.astype(int)] = 1
                labels = one_hot_labels
            elif labels.shape[1] != self.n_classes:
                raise ValueError(f"Labels must have {self.n_classes} columns when one-hot encoded")

        latent_points, _ = self._generate_latent_points(n_samples, labels)

        samples = self.generator.forward_pass(latent_points, training=False)
        return samples

    def evaluate(
       self,
       x_test: np.ndarray,
       y_test: np.ndarray | None = None,
       batch_size: int = 32,
       n_gen_samples: int | None = None
   ) -> tuple[float, float]:

       if batch_size is None:
           batch_size = len(x_test)

       total_d_loss = 0
       total_g_loss = 0
       n_batches = 0

       for start_idx in range(0, len(x_test), batch_size):
           end_idx = min(start_idx + batch_size, len(x_test))
           current_batch_size = end_idx - start_idx

           x_batch = x_test[start_idx:end_idx]
           batch_labels = y_test[start_idx:end_idx] if y_test is not None else None

           latent_points, gen_labels = self._generate_latent_points(current_batch_size, batch_labels)
           fake_samples = self.generator.forward_pass(latent_points, training=False)

           y_real = np.ones((current_batch_size, 1))
           y_fake = np.zeros((current_batch_size, 1))

           if self.n_classes is not None and batch_labels is not None:
               discriminator_real_input = np.concatenate([x_batch, gen_labels], axis=1)
               discriminator_fake_input = np.concatenate([fake_samples, gen_labels], axis=1)
           else:
               discriminator_real_input = x_batch
               discriminator_fake_input = fake_samples

           d_loss_real = self.discriminator_loss(
               y_real,
               self.discriminator.forward_pass(discriminator_real_input, training=False)
           )
           d_loss_fake = self.discriminator_loss(
               y_fake,
               self.discriminator.forward_pass(discriminator_fake_input, training=False)
           )
           d_loss = 0.5 * (d_loss_real + d_loss_fake)

           y_gan = np.ones((current_batch_size, 1))
           latent_points, gen_labels = self._generate_latent_points(current_batch_size, batch_labels)
           fake_samples = self.generator.forward_pass(latent_points, training=False)

           if self.n_classes is not None:
               discriminator_gen_input = np.concatenate([fake_samples, gen_labels], axis=1)
           else:
               discriminator_gen_input = fake_samples

           g_loss = self.generator_loss(
               y_gan,
               self.discriminator.forward_pass(discriminator_gen_input, training=False)
           )

           total_d_loss += d_loss * current_batch_size
           total_g_loss += g_loss * current_batch_size
           n_batches += current_batch_size

       avg_d_loss = total_d_loss / n_batches
       avg_g_loss = total_g_loss / n_batches

       return float(avg_d_loss), float(avg_g_loss)

    def save(self, filename: str):
        base, ext = os.path.splitext(filename)

        model_state = {
            'type': 'GAN',
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state,
            'use_spectral_norm': self.use_spectral_norm,
            'use_gradient_penalty': self.use_gradient_penalty,
            'gp_weight': self.gp_weight,
            'image_height': self._image_height,
            'image_width': self._image_width,
            'label_smoothing': self.label_smoothing,
            'generator': self.generator.save(f"{base}_generator{ext}") if self.generator else None,
            'discriminator': self.discriminator.save(f"{base}_discriminator{ext}") if self.discriminator else None,
            'generator_optimizer': self.generator_optimizer.get_config() if self.generator_optimizer else None,
            'discriminator_optimizer': self.discriminator_optimizer.get_config() if self.discriminator_optimizer else None,
            'generator_loss': self.generator_loss.get_config() if self.generator_loss else None,
            'discriminator_loss': self.discriminator_loss.get_config() if self.discriminator_loss else None
        }

        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4, default=to_json_serializable)

    @classmethod
    def load(cls, filename: str) -> 'GAN':
        base, ext = os.path.splitext(filename)

        with open(filename, 'r') as f:
            model_state = json.load(f)

        model = cls(
            latent_dim=model_state['latent_dim'],
            n_classes=model_state.get('n_classes'),
            gradient_clip_threshold=model_state['gradient_clip_threshold'],
            enable_padding=model_state['enable_padding'],
            padding_size=model_state['padding_size'],
            random_state=model_state['random_state'],
            use_spectral_norm=model_state.get('use_spectral_norm', True),
            use_gradient_penalty=model_state.get('use_gradient_penalty', True),
            gp_weight=model_state.get('gp_weight', 10.0),
            image_height=model_state.get('image_height'),
            image_width=model_state.get('image_width'),
            label_smoothing=model_state.get('label_smoothing', 0.9)
        )

        model.generator = Sequential.load(f"{base}_generator{ext}")

        model.discriminator = Sequential.load(f"{base}_discriminator{ext}")

        if model_state.get('generator_optimizer'):
            model.generator_optimizer = Optimizer.from_config(
                model_state['generator_optimizer'])

        if model_state.get('discriminator_optimizer'):
            model.discriminator_optimizer = Optimizer.from_config(
                model_state['discriminator_optimizer'])

        if model_state.get('generator_loss'):
            model.generator_loss = LossFunction.from_config(
                model_state['generator_loss'])

        if model_state.get('discriminator_loss'):
            model.discriminator_loss = LossFunction.from_config(
                model_state['discriminator_loss'])

        # the sub-models must use the restored optimizers and losses (as after compile)
        model.generator.optimizer = model.generator_optimizer or model.generator.optimizer
        model.generator.loss_function = model.generator_loss or model.generator.loss_function
        model.discriminator.optimizer = model.discriminator_optimizer or model.discriminator.optimizer
        model.discriminator.loss_function = model.discriminator_loss or model.discriminator.loss_function

        return model

    def __str__(self) -> str:
        model_summary = (
            f'GAN(latent_dim={self.latent_dim}, '
            f'gradient_clip_threshold={self.gradient_clip_threshold}, '
            f'enable_padding={self.enable_padding}, '
            f'padding_size={self.padding_size}, '
            f'random_state={self.random_state})\n'
        )
        model_summary += '-------------------------------------------------\n'
        model_summary += 'Generator:\n'
        model_summary += str(self.generator) if self.generator else "Not compiled yet\n"
        model_summary += '-------------------------------------------------\n'
        model_summary += 'Discriminator:\n'
        model_summary += str(
            self.discriminator) if self.discriminator else "Not compiled yet\n"
        return model_summary

    def save_weights(self, epoch: int):
        weights = {
            'generator': [layer.weights.copy() if getattr(layer, 'weights', None) is not None else None
                          for layer in self.generator.layers],
            'biases': [layer.bias.copy() if getattr(layer, 'bias', None) is not None else None
                       for layer in self.generator.layers],
            'epoch': epoch
        }
        return weights

    def load_weights(self, weights):
        for layer, w, b in zip(self.generator.layers, weights['generator'], weights['biases']):
            if w is not None and hasattr(layer, 'weights'):
                layer.weights = w.copy()
            if b is not None and hasattr(layer, 'bias'):
                layer.bias = b.copy()
