import os
import inspect
import json
import time
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.callbacks import EarlyStopping
from neuralnetlib.layers import *
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy
from neuralnetlib.metrics import Metric
from neuralnetlib.optimizers import Optimizer
from neuralnetlib.preprocessing import PCA, pad_sequences, clip_gradients, SpectralNorm
from neuralnetlib.utils import shuffle, progress_bar, is_interactive, is_display_available, format_number, log_softmax, train_test_split, History, GradientDebugger


class BaseModel(ABC):
    def __init__(self, gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None):

        self.gradient_clip_threshold = gradient_clip_threshold
        self.enable_padding = enable_padding
        self.padding_size = padding_size
        self.random_state = random_state if random_state is not None else time.time_ns()

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
    def __init__(self, gradient_clip_threshold: float = 5.0,
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

    def add(self, layer: Layer):
        if not self.layers:
            if not isinstance(layer, Input):
                raise ValueError("The first layer must be an Input layer.")
        else:
            if self.random_state and hasattr(layer, 'random_state'):
                layer.random_state = self.random_state
                
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

        activation_attr = getattr(layer, 'activation', getattr(
            layer, 'activation_function', None))
        if activation_attr and not isinstance(layer, Activation):
            if isinstance(activation_attr, str):
                activation = Activation.from_name(activation_attr)
            elif isinstance(activation_attr, ActivationFunction):
                activation = Activation(activation_attr)
            elif isinstance(activation_attr, Activation):
                activation = activation_attr
            else:
                raise ValueError(
                    f"Invalid activation function: {activation_attr}")
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

        if self.enable_padding:
            original_shape = X.shape
            padded_shape = ((original_shape[0] + self.padding_size - 1) //
                            self.padding_size * self.padding_size,) + original_shape[1:]

            if padded_shape != original_shape:
                padded_X = np.zeros(padded_shape, dtype=X.dtype)
                padded_X[:original_shape[0]] = X
                X = padded_X

        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization, LSTM, Bidirectional, GRU)):
                X = layer.forward_pass(X, training)
            else:
                X = layer.forward_pass(X)

        if self.enable_padding and padded_shape != original_shape:
            X = X[:original_shape[0]]

        self.predictions = X
        return X

    def backward_pass(self, error: np.ndarray, gan: bool = False, compute_only: bool = False) -> np.ndarray:
        
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0 and isinstance(layer, Activation):
                if not gan:
                    if (type(layer.activation_function).__name__ == "Softmax" and
                            isinstance(self.loss_function, CategoricalCrossentropy)):
                        error = self.predictions - self.y_true
                    elif (type(layer.activation_function).__name__ == "Sigmoid" and
                          isinstance(self.loss_function, BinaryCrossentropy)):
                        error = self.predictions - self.y_true
                    elif isinstance(self.loss_function, SparseCategoricalCrossentropy):
                        y_true_one_hot = np.zeros_like(self.predictions)
                        y_true_one_hot[np.arange(
                            len(self.y_true)), self.y_true] = 1
                        error = self.predictions - y_true_one_hot
                else:
                    error = layer.backward_pass(error)
            else:
                error = clip_gradients(error)
                error = layer.backward_pass(error)

            if compute_only:
                continue

            layer_idx = len(self.layers) - 1 - i

            if isinstance(layer, LSTM):
                cell = layer.cell
                for grad_pair in [(cell.dWf, cell.dbf), (cell.dWi, cell.dbi),
                                  (cell.dWc, cell.dbc), (cell.dWo, cell.dbo)]:
                    weight_grad, bias_grad = grad_pair
                    self.optimizer.update(layer_idx, grad_pair[0], clip_gradients(weight_grad),
                                          grad_pair[1], clip_gradients(bias_grad))

            elif isinstance(layer, GRU):
                cell = layer.cell
                self.optimizer.update(layer_idx, cell.Wz, clip_gradients(cell.dWz),
                                      cell.bz, clip_gradients(cell.dbz))
                self.optimizer.update(layer_idx, cell.Wr, clip_gradients(cell.dWr),
                                      cell.br, clip_gradients(cell.dbr))
                self.optimizer.update(layer_idx, cell.Wh, clip_gradients(cell.dWh),
                                      cell.bh, clip_gradients(cell.dbh))

            elif hasattr(layer, 'weights'):
                clipped_weights_grad = clip_gradients(layer.d_weights)
                if hasattr(layer, 'd_bias'):
                    clipped_bias_grad = clip_gradients(layer.d_bias)
                    self.optimizer.update(layer_idx, layer.weights, clipped_weights_grad,
                                          layer.bias, clipped_bias_grad)
                else:
                    self.optimizer.update(
                        layer_idx, layer.weights, clipped_weights_grad)

        return error

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        self.y_true = y_batch
        self.predictions = self.forward_pass(x_batch)
        predictions = self.predictions.copy()
        loss = self.loss_function(y_batch, predictions)
        error = self.loss_function.derivative(y_batch, predictions)

        if error.ndim == 1:
            error = error[:, None]
        elif isinstance(self.layers[-1], (LSTM, Bidirectional, GRU)) and self.layers[-1].return_sequences:
            error = error.reshape(error.shape[0], error.shape[1], -1)

        self.backward_pass(error)
        return loss

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
                random_state=random_state if random_state is not None else self.random_state
            )
            validation_data = (x_val, y_val)

        if plot_decision_boundary and not is_interactive() and not is_display_available():
            raise ValueError("Cannot display the plot. Please run the script in an environment with a display.")

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        for layer in self.layers:
            if hasattr(layer, 'random_state'):
                layer.random_state = random_state if random_state is not None else self.random_state

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU)) for layer in self.layers)
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

        if metrics is not None:
            processed_metrics = []
            for metric in metrics:
                if metric == 'val_loss':
                    continue
                try:
                    metric_obj = Metric(metric)
                    processed_metrics.append(metric_obj)
                    history[metric_obj.name] = []
                    if validation_data is not None:
                        history[f'val_{metric_obj.name}'] = []
                except ValueError as e:
                    if metric not in ['val_loss']:
                        raise ValueError(f"Invalid metric: {metric}") from e
            metrics = processed_metrics

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

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()
                x_train_shuffled, y_train_shuffled = shuffle(
                    x_train, y_train,
                    random_state=random_state if random_state is not None else self.random_state
                )

                error = 0
                predictions_list = []
                y_true_list = []

                if batch_size is not None:
                    num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)
                    
                    for j in range(0, x_train.shape[0], batch_size):
                        batch_index = j // batch_size
                        
                        x_batch = x_train_shuffled[j:j + batch_size]
                        y_batch = y_train_shuffled[j:j + batch_size]
                        if y_batch.ndim == 1:
                            y_batch = y_batch.reshape(-1, 1)

                        batch_logs = {
                            'batch': batch_index,
                            'size': len(x_batch),
                            'model': self
                        }
                        for callback in callbacks:
                            callback.on_batch_begin(batch_index, batch_logs)

                        batch_error = self.train_on_batch(x_batch, y_batch)
                        error += batch_error
                        predictions_list.append(self.predictions)
                        y_true_list.append(y_batch)

                        batch_logs.update({'loss': batch_error})
                        
                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(
                                    np.vstack(predictions_list[-1:]), 
                                    np.vstack(y_true_list[-1:])
                                )
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
                                        np.vstack(predictions_list), 
                                        np.vstack(y_true_list)
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
                    error = self.train_on_batch(x_train, y_train)
                    predictions_list.append(self.predictions)
                    y_true_list.append(y_train)

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
                                    np.vstack(predictions_list), 
                                    np.vstack(y_true_list)
                                )
                                history[metric.name].append(metric_value)
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
                            np.vstack(predictions_list), 
                            np.vstack(y_true_list)
                        )
                        history[metric.name].append(metric_value)
                        epoch_logs[metric.name] = metric_value

                if validation_data is not None:
                    val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

                    if metrics is not None:
                        val_metrics = []
                        for metric in metrics:
                            val_metric = metric(val_predictions, y_val)
                            history[f'val_{metric.name}'].append(val_metric)
                            epoch_logs[f'val_{metric.name}'] = val_metric
                            val_metrics.append(val_metric)

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
                        random_state if random_state is not None else self.random_state
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
        total_loss = 0
        num_batches = int(np.ceil(len(x_test) / batch_size))

        predictions_list = []

        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]

            batch_predictions = self.forward_pass(batch_x, training=False)
            batch_loss = self.loss_function(batch_y, batch_predictions)

            total_loss += batch_loss
            predictions_list.append(batch_predictions)

            for layer in self.layers:
                if hasattr(layer, 'reset_cache'):
                    layer.reset_cache()

        avg_loss = total_loss / num_batches

        all_predictions = np.vstack(predictions_list)
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

        for _ in range(max_length - sequence_start.shape[1]):
            # cuz we already apply temperature in this method
            predictions = self.predict(current_sequence)

            if predictions.ndim == 3:
                next_token_probs = predictions[:, -1, :]
            else:
                next_token_probs = predictions

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

            rng = np.random.default_rng(self.random_state)

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

            self.random_state += 1

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
            json.dump(model_state, f, indent=4)
            
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

        if y_train.ndim > 1:
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
        
        self.latent_dim = None

    def _calculate_kl_divergence(self):
        if not self.variational:
            return 0.0
        kl_loss = -0.5 * np.mean(
            1 + self.latent_log_var -
            np.square(self.latent_mean) - np.exp(self.latent_log_var)
        )
        return kl_loss

    def _reparameterize(self):
        if not self.variational:
            return self.latent_space
        rng = np.random.default_rng(self.random_state)
        epsilon = rng.normal(size=self.latent_mean.shape)
        return self.latent_mean + np.exp(0.5 * self.latent_log_var) * epsilon

    def _calculate_regularization(self):
        reg_loss = 0.0

        def process_layer(layer):
            reg = 0.0
            if hasattr(layer, 'weights'):
                if self.l1_reg > 0:
                    reg += self.l1_reg * np.sum(np.abs(layer.weights))
                if self.l2_reg > 0:
                    reg += self.l2_reg * np.sum(np.square(layer.weights))
            return reg

        for layer in self.encoder_layers:
            reg_loss += process_layer(layer)

        for layer in self.decoder_layers:
            reg_loss += process_layer(layer)

        return reg_loss

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

        self.encoder_layers.append(layer)

        activation_attr = getattr(layer, 'activation', getattr(
            layer, 'activation_function', None))
        if activation_attr and not isinstance(layer, Activation):
            if isinstance(activation_attr, str):
                activation = Activation.from_name(activation_attr)
            elif isinstance(activation_attr, ActivationFunction):
                activation = Activation(activation_attr)
            elif isinstance(activation_attr, Activation):
                activation = activation_attr
            else:
                raise ValueError(
                    f"Invalid activation function: {activation_attr}")
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

        self.decoder_layers.append(layer)

        activation_attr = getattr(layer, 'activation', getattr(
            layer, 'activation_function', None))
        if activation_attr and not isinstance(layer, Activation):
            if isinstance(activation_attr, str):
                activation = Activation.from_name(activation_attr)
            elif isinstance(activation_attr, ActivationFunction):
                activation = Activation(activation_attr)
            elif isinstance(activation_attr, Activation):
                activation = activation_attr
            else:
                raise ValueError(
                    f"Invalid activation function: {activation_attr}")
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
            
        self.latent_dim = self.encoder_layers[-1].units if self.variational else self.encoder_layers[-1].output_shape[-1]

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if self.enable_padding:
            original_shape = X.shape
            padded_shape = ((original_shape[0] + self.padding_size - 1) //
                            self.padding_size * self.padding_size,) + original_shape[1:]

            if padded_shape != original_shape:
                padded_X = np.zeros(padded_shape, dtype=X.dtype)
                padded_X[:original_shape[0]] = X
                X = padded_X

        self.encoder_activations = []
        self.decoder_activations = []
        self.skip_cache = {}

        # Encoder forward pass
        encoded = X
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, (Dropout, LSTM, Bidirectional, GRU)):
                encoded = layer.forward_pass(encoded, training)
            else:
                encoded = layer.forward_pass(encoded)
            self.encoder_activations.append(encoded)

            if self.skip_connections and isinstance(layer, Dense):
                self.skip_cache[layer.units] = encoded

        if self.variational:
            latent_dim = encoded.shape[-1] // 2
            self.latent_mean = encoded[:, :latent_dim]
            self.latent_log_var = encoded[:, latent_dim:]
            self.latent_space = self._reparameterize()
        else:
            self.latent_space = encoded

        # Decoder forward pass
        decoded = encoded

        for layer in self.decoder_layers:
            if isinstance(layer, (Dropout, LSTM, Bidirectional, GRU)):
                decoded = layer.forward_pass(decoded, training)
            else:
                decoded = layer.forward_pass(decoded)
            if self.skip_connections and isinstance(layer, Dense):
                skip_connection = self.skip_cache.get(layer.units)
                if skip_connection is not None:
                    scale_factor = 1.0 / np.sqrt(layer.units)
                    decoded = decoded + scale_factor * skip_connection

            self.decoder_activations.append(decoded)

        if self.enable_padding and padded_shape != original_shape:
            decoded = decoded[:original_shape[0]]

        return decoded

    def backward_pass(self, error: np.ndarray):
        def clip_gradients(gradient: np.ndarray) -> np.ndarray:
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

        # Decoder backward pass
        for i, layer in enumerate(reversed(self.decoder_layers)):
            if i == 0 and isinstance(layer, Activation):
                if (type(layer.activation_function).__name__ == "Softmax" and
                        isinstance(self.decoder_loss, CategoricalCrossentropy)):
                    error = self.predictions - self.y_true
                elif (type(layer.activation_function).__name__ == "Sigmoid" and
                      isinstance(self.decoder_loss, BinaryCrossentropy)):
                    error = (self.predictions - self.y_true) / (self.predictions *
                                                                (1 - self.predictions) + 1e-15)
            else:
                error = clip_gradients(error)
                error = layer.backward_pass(error)

            layer_idx = len(self.decoder_layers) - 1 - i

            if isinstance(layer, (LSTM, GRU)):
                self._update_rnn_weights(
                    layer, layer_idx, clip_gradients, self.decoder_optimizer)
            elif hasattr(layer, 'weights'):
                self._update_layer_weights(
                    layer, layer_idx, clip_gradients, self.decoder_optimizer)

        # Encoder backward pass
        for i, layer in enumerate(reversed(self.encoder_layers)):
            error = clip_gradients(error)
            error = layer.backward_pass(error)

            layer_idx = len(self.encoder_layers) - 1 - i

            if isinstance(layer, (LSTM, GRU)):
                self._update_rnn_weights(
                    layer, layer_idx, clip_gradients, self.encoder_optimizer)
            elif hasattr(layer, 'weights'):
                self._update_layer_weights(
                    layer, layer_idx, clip_gradients, self.encoder_optimizer)

    def _update_rnn_weights(self, layer, layer_idx: int, clip_gradients, optimizer):
        if isinstance(layer, LSTM):
            cell = layer.cell
            for grad_pair in [(cell.dWf, cell.dbf), (cell.dWi, cell.dbi),
                              (cell.dWc, cell.dbc), (cell.dWo, cell.dbo)]:
                weight_grad, bias_grad = grad_pair
                clipped_weight_grad = clip_gradients(weight_grad)
                clipped_bias_grad = clip_gradients(bias_grad)
                optimizer.update(layer_idx, cell.Wf, clipped_weight_grad,
                                 cell.bf, clipped_bias_grad)

        elif isinstance(layer, GRU):
            cell = layer.cell
            optimizer.update(layer_idx, cell.Wz, clip_gradients(cell.dWz),
                             cell.bz, clip_gradients(cell.dbz))
            optimizer.update(layer_idx, cell.Wr, clip_gradients(cell.dWr),
                             cell.br, clip_gradients(cell.dbr))
            optimizer.update(layer_idx, cell.Wh, clip_gradients(cell.dWh),
                             cell.bh, clip_gradients(cell.dbh))

    def _update_layer_weights(self, layer, layer_idx: int, clip_gradients, optimizer):
        clipped_weights_grad = clip_gradients(layer.d_weights)
        if hasattr(layer, 'd_bias'):
            clipped_bias_grad = clip_gradients(layer.d_bias)
            optimizer.update(layer_idx, layer.weights, clipped_weights_grad,
                             layer.bias, clipped_bias_grad)
        else:
            optimizer.update(layer_idx, layer.weights, clipped_weights_grad)

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray = None) -> float:
        if y_batch is None:
            y_batch = x_batch

        self.y_true = y_batch
        self.predictions = self.forward_pass(x_batch)

        reconstruction_loss = self.decoder_loss(y_batch, self.predictions)
        regularization_loss = self._calculate_regularization()
        kl_loss = self._calculate_kl_divergence() if self.variational else 0

        latent_l2 = 0.0001 * np.mean(np.square(self.latent_space))
        latent_std = np.std(self.latent_space, axis=0)
        distribution_penalty = 0.0001 * np.mean(np.abs(latent_std - 1.0))

        if self.skip_connections:
            latent_l2 *= 0.1
            distribution_penalty *= 0.1

        beta = 0.01
        total_loss = reconstruction_loss + regularization_loss + \
            latent_l2 + distribution_penalty + beta * kl_loss

        error = self.decoder_loss.derivative(y_batch, self.predictions)
        if error.ndim == 1:
            error = error[:, None]
        elif isinstance(self.decoder_layers[-1], (LSTM, Bidirectional, GRU)) and self.decoder_layers[-1].return_sequences:
            error = error.reshape(error.shape[0], error.shape[1], -1)

        self.backward_pass(error)
        return total_loss

    def predict(self, X: np.ndarray, output_latent: bool = False, temperature: float = 1.0) -> np.ndarray:
        X = np.array(X)
        encoded = X
        for layer in self.encoder_layers:
            encoded = layer.forward_pass(encoded)

        if output_latent:
            return encoded

        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer.forward_pass(decoded)

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

        total_loss = 0
        num_batches = int(np.ceil(len(x_test) / batch_size))
        predictions_list = []

        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]

            batch_predictions = self.forward_pass(batch_x, training=False)
            decoder_loss = self.decoder_loss(batch_y, batch_predictions)
            encoder_loss = self.encoder_loss(batch_y, batch_predictions)
            batch_loss = (decoder_loss + encoder_loss) / 2

            total_loss += batch_loss
            predictions_list.append(batch_predictions)

        avg_loss = total_loss / num_batches
        all_predictions = np.vstack(predictions_list)

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
            'variational': self.variational
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
            json.dump(model_state, f, indent=4)

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
                random_state=random_state
            )
            validation_data = (x_val, x_val)

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train

        for layer in self.encoder_layers + self.decoder_layers:
            if hasattr(layer, 'random_state'):
                layer.random_state = random_state if random_state is not None else self.random_state

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU))
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

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()
                
                y_train = np.zeros_like(x_train)
                x_train_shuffled, _ = shuffle(
                    x_train, y_train,
                    random_state=random_state if random_state is not None else self.random_state
                )

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
                        predictions_list.append(self.predictions)
                        inputs_list.append(x_batch)

                        batch_logs.update({
                            'loss': batch_error,
                        })

                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(
                                    np.vstack(predictions_list[-1:]),
                                    np.vstack(inputs_list[-1:])
                                )
                                batch_metrics[metric.name] = batch_metric_value
                            batch_logs.update(batch_metrics)

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        if verbose:
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metric_value = metric(
                                        np.vstack(predictions_list),
                                        np.vstack(inputs_list)
                                    )
                                    metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                            progress_bar(
                                batch_index + 1,
                                num_batches,
                                message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error / (batch_index + 1))} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s'
                            )

                    error /= num_batches

                else:
                    error = self.train_on_batch(x_train)
                    predictions_list.append(self.predictions)
                    inputs_list.append(x_train)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(
                                    np.vstack(predictions_list),
                                    np.vstack(inputs_list)
                                )
                                history[metric.name].append(metric_value)
                                metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                        progress_bar(
                            1, 1,
                            message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error)} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s'
                        )

                history['loss'].append(error)

                epoch_logs.update({
                    'loss': error,
                    'time': time.time() - start_time
                })

                if metrics is not None:
                    for metric in metrics:
                        metric_value = metric(
                            np.vstack(predictions_list),
                            np.vstack(inputs_list)
                        )
                        epoch_logs[metric.name] = metric_value
                        
                if validation_data is not None:
                    val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

                    if metrics is not None:
                        val_metrics = []
                        for metric in metrics:
                            val_metric = metric(val_predictions, x_val)
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
        _ = self.forward_pass(x_train[:n_examples])

        latent_mean_stats = np.mean(self.latent_mean, axis=0)
        latent_std_stats = np.exp(0.5 * np.mean(self.latent_log_var, axis=0))

        latent_mean_repeated = np.tile(latent_mean_stats, (n_samples, 1))
        latent_std_repeated = np.tile(latent_std_stats, (n_samples, 1))

        rng = np.random.default_rng(
            seed if seed is not None else self.random_state)
        noise = rng.standard_normal(size=(n_samples, self.latent_dim))

        latent_samples = np.concatenate([
            latent_mean_repeated + noise *
            latent_std_repeated, np.zeros((n_samples, self.latent_dim))
        ], axis=1)

        generated = latent_samples
        for layer in self.decoder_layers:
            if isinstance(layer, (Dropout, LSTM, Bidirectional, GRU)):
                generated = layer.forward_pass(generated, training=False)
            elif isinstance(layer, Reshape):
                if isinstance(layer.target_shape, list):
                    layer.target_shape = tuple(layer.target_shape)
                generated = layer.forward_pass(generated)
            else:
                generated = layer.forward_pass(generated)

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
        self.gradient_norms = {}

        self.src_embedding = Embedding(
            self.src_vocab_size, d_model, input_length=max_sequence_length, random_state=random_state)
        self.tgt_embedding = Embedding(
            self.tgt_vocab_size, d_model, input_length=max_sequence_length, random_state=random_state)

        self.positional_encoding = PositionalEncoding(
            max_sequence_length=max_sequence_length,
            embedding_dim=d_model,
            scale_embeddings=scale_embeddings
        )

        self.encoder_dropout = Dropout(dropout_rate, random_state=random_state)
        self.encoder_layers: list = []
        for _ in range(n_encoder_layers):
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                attention_dropout=dropout_rate,
                random_state=random_state,
            )
            self.encoder_layers.append(encoder_layer)

        self.decoder_dropout = Dropout(dropout_rate, random_state=random_state)
        self.decoder_layers: list = []
        for _ in range(n_decoder_layers):
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                attention_dropout=dropout_rate,
                random_state=random_state
            )
            self.decoder_layers.append(decoder_layer)

        self.output_layer = Dense(tgt_vocab_size, random_state=random_state)

        self.optimizer = None
        self.loss_function = None

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
        x = self.positional_encoding.forward_pass(x)
        x = self.encoder_dropout.forward_pass(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward_pass(x, mask=mask, training=training)

        return x

    def decode(self, tar: np.ndarray, enc_output: np.ndarray, training: bool = True,
               look_ahead_mask: np.ndarray | None = None,
               padding_mask: np.ndarray | None = None) -> np.ndarray:

        x = self.tgt_embedding.forward_pass(tar)
        x = self.positional_encoding.forward_pass(x)

        if training:
            x = self.decoder_dropout.forward_pass(x, training=True)

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
        error = clip_gradients(error)

        dx = self.output_layer.backward_pass(error)
        d_enc_output = None

        for decoder_layer in reversed(self.decoder_layers):
            dx, d_enc = decoder_layer.backward_pass(dx)
            d_enc_output = d_enc if d_enc_output is None else d_enc_output + d_enc

        dx = self.decoder_dropout.backward_pass(dx)
        dx = self.tgt_embedding.backward_pass(dx)
        dx = self.positional_encoding.backward_pass(dx)

        dx_enc = d_enc_output
        for encoder_layer in reversed(self.encoder_layers):
            dx_enc = encoder_layer.backward_pass(dx_enc)

        dx_enc = self.encoder_dropout.backward_pass(dx_enc)
        dx_enc = self.src_embedding.backward_pass(dx_enc)

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

        decoder_input = pad_sequences(decoder_input,
                                      max_length=self.max_sequence_length,
                                      padding='post',
                                      pad_value=self.PAD_IDX)

        decoder_target = pad_sequences(decoder_target,
                                       max_length=self.max_sequence_length,
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

    def train_on_batch(self, x_batch: tuple[np.ndarray, np.ndarray], y_batch: np.ndarray, print_logging: bool = False) -> float:
        self.gradient_norms = {}

        if isinstance(x_batch, list) and len(x_batch) == 2:
            encoder_input, decoder_input = x_batch
        else:
            raise ValueError(
                "x_batch must be a list of [encoder_input, decoder_input]")

        decoder_target = y_batch

        self.enc_seq_length = encoder_input.shape[1]
        self.dec_seq_length = decoder_input.shape[1]

        self.predictions = self.forward_pass(
            (encoder_input, decoder_input), training=True)

        loss = self.loss_function(decoder_target, self.predictions)
        error = self.loss_function.derivative(decoder_target, self.predictions)
        self.backward_pass(error)

        def update_with_monitoring(name: str, layer_idx: int, weights: np.ndarray,
                                   d_weights: np.ndarray, bias: np.ndarray, d_bias: np.ndarray) -> None:
            weight_norm = np.linalg.norm(d_weights)
            bias_norm = np.linalg.norm(d_bias) if d_bias is not None else 0

            self.gradient_norms[name] = {
                'weights': float(weight_norm),
                'bias': float(bias_norm)
            }

            if print_logging and weight_norm > self.gradient_clip_threshold:
                logging.warning(
                    f"Large gradient norm in {name}: {weight_norm:.4f}")

            self.optimizer.update(layer_idx, weights, d_weights, bias, d_bias)

        layer_idx = 0

        update_with_monitoring(
            'src_embedding', layer_idx,
            self.src_embedding.weights, self.src_embedding.d_weights,
            self.src_embedding.bias, self.src_embedding.d_bias
        )
        layer_idx += 1

        update_with_monitoring(
            'tgt_embedding', layer_idx,
            self.tgt_embedding.weights, self.tgt_embedding.d_weights,
            self.tgt_embedding.bias, self.tgt_embedding.d_bias
        )
        layer_idx += 1

        for i, encoder_layer in enumerate(self.encoder_layers):
            update_with_monitoring(
                f'encoder_{i}_self_attn_Q', layer_idx,
                encoder_layer.attention.query_dense.weights,
                encoder_layer.attention.query_dense.d_weights,
                encoder_layer.attention.query_dense.bias,
                encoder_layer.attention.query_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'encoder_{i}_self_attn_K', layer_idx,
                encoder_layer.attention.key_dense.weights,
                encoder_layer.attention.key_dense.d_weights,
                encoder_layer.attention.key_dense.bias,
                encoder_layer.attention.key_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'encoder_{i}_self_attn_V', layer_idx,
                encoder_layer.attention.value_dense.weights,
                encoder_layer.attention.value_dense.d_weights,
                encoder_layer.attention.value_dense.bias,
                encoder_layer.attention.value_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'encoder_{i}_self_attn_O', layer_idx,
                encoder_layer.attention.output_dense.weights,
                encoder_layer.attention.output_dense.d_weights,
                encoder_layer.attention.output_dense.bias,
                encoder_layer.attention.output_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'encoder_{i}_ffn_1', layer_idx,
                encoder_layer.ffn.dense1.weights,
                encoder_layer.ffn.dense1.d_weights,
                encoder_layer.ffn.dense1.bias,
                encoder_layer.ffn.dense1.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'encoder_{i}_ffn_2', layer_idx,
                encoder_layer.ffn.dense2.weights,
                encoder_layer.ffn.dense2.d_weights,
                encoder_layer.ffn.dense2.bias,
                encoder_layer.ffn.dense2.d_bias
            )
            layer_idx += 1

        for i, decoder_layer in enumerate(self.decoder_layers):
            update_with_monitoring(
                f'decoder_{i}_self_attn_Q', layer_idx,
                decoder_layer.self_attention.query_dense.weights,
                decoder_layer.self_attention.query_dense.d_weights,
                decoder_layer.self_attention.query_dense.bias,
                decoder_layer.self_attention.query_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_self_attn_K', layer_idx,
                decoder_layer.self_attention.key_dense.weights,
                decoder_layer.self_attention.key_dense.d_weights,
                decoder_layer.self_attention.key_dense.bias,
                decoder_layer.self_attention.key_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_self_attn_V', layer_idx,
                decoder_layer.self_attention.value_dense.weights,
                decoder_layer.self_attention.value_dense.d_weights,
                decoder_layer.self_attention.value_dense.bias,
                decoder_layer.self_attention.value_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_self_attn_O', layer_idx,
                decoder_layer.self_attention.output_dense.weights,
                decoder_layer.self_attention.output_dense.d_weights,
                decoder_layer.self_attention.output_dense.bias,
                decoder_layer.self_attention.output_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_cross_attn_Q', layer_idx,
                decoder_layer.cross_attention.query_dense.weights,
                decoder_layer.cross_attention.query_dense.d_weights,
                decoder_layer.cross_attention.query_dense.bias,
                decoder_layer.cross_attention.query_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_cross_attn_K', layer_idx,
                decoder_layer.cross_attention.key_dense.weights,
                decoder_layer.cross_attention.key_dense.d_weights,
                decoder_layer.cross_attention.key_dense.bias,
                decoder_layer.cross_attention.key_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_cross_attn_V', layer_idx,
                decoder_layer.cross_attention.value_dense.weights,
                decoder_layer.cross_attention.value_dense.d_weights,
                decoder_layer.cross_attention.value_dense.bias,
                decoder_layer.cross_attention.value_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_cross_attn_O', layer_idx,
                decoder_layer.cross_attention.output_dense.weights,
                decoder_layer.cross_attention.output_dense.d_weights,
                decoder_layer.cross_attention.output_dense.bias,
                decoder_layer.cross_attention.output_dense.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_ffn_1', layer_idx,
                decoder_layer.ffn.dense1.weights,
                decoder_layer.ffn.dense1.d_weights,
                decoder_layer.ffn.dense1.bias,
                decoder_layer.ffn.dense1.d_bias
            )
            layer_idx += 1

            update_with_monitoring(
                f'decoder_{i}_ffn_2', layer_idx,
                decoder_layer.ffn.dense2.weights,
                decoder_layer.ffn.dense2.d_weights,
                decoder_layer.ffn.dense2.bias,
                decoder_layer.ffn.dense2.d_bias
            )
            layer_idx += 1

        update_with_monitoring(
            'output', layer_idx,
            self.output_layer.weights,
            self.output_layer.d_weights,
            self.output_layer.bias,
            self.output_layer.d_bias
        )

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
            
        history = History({
            'loss': [],
            'val_loss': []
        })

        if validation_data is not None and validation_split is not None:
            raise ValueError("Cannot specify both validation_data and validation_split")
        elif validation_data is None and validation_split is not None:
            x_val, y_val = train_test_split(
                x_train, test_size=validation_split, random_state=random_state)
            validation_data = (x_val, y_val)

        encoder_input, decoder_input, decoder_target = self.prepare_data(x_train, y_train)

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

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()

                indices = np.arange(len(encoder_input))
                if random_state is not None:
                    rng = np.random.default_rng(
                        random_state if random_state is not None else self.random_state)
                    rng.shuffle(indices)
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
                        predictions_list.append(self.predictions)
                        y_true_list.append(target_batch)

                        batch_logs.update({'loss': batch_error})
                        
                        if metrics is not None:
                            batch_metrics = {}
                            for metric in metrics:
                                batch_metric_value = metric(np.vstack(predictions_list[-1:]), 
                                                        np.vstack(y_true_list[-1:]))
                                batch_metrics[metric.name] = batch_metric_value
                            batch_logs.update(batch_metrics)

                        for callback in callbacks:
                            callback.on_batch_end(batch_index, batch_logs)

                        if verbose:
                            metrics_str = ''
                            if metrics is not None:
                                for metric in metrics:
                                    metric_value = metric(np.vstack(predictions_list), 
                                                        np.vstack(y_true_list))
                                    metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                            progress_bar(batch_index + 1, num_batches,
                                    message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error / (batch_index + 1))} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

                    error /= num_batches
                else:
                    error = self.train_on_batch([encoder_input, decoder_input], decoder_target)
                    predictions_list.append(self.predictions)
                    y_true_list.append(decoder_target)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(np.vstack(predictions_list), 
                                                np.vstack(y_true_list))
                                history[metric.name].append(metric_value)
                                metrics_str += f'{metric.name}: {format_number(metric_value)} - '
                        progress_bar(1, 1,
                                message=f'Epoch {epoch + 1}/{epochs} - loss: {format_number(error)} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

                history['loss'].append(error)

                epoch_logs.update({
                    'loss': error,
                    'time': time.time() - start_time
                })

                if metrics is not None:
                    for metric in metrics:
                        metric_value = metric(np.vstack(predictions_list), 
                                        np.vstack(y_true_list))
                        epoch_logs[metric.name] = metric_value

                if validation_data is not None:
                    if isinstance(validation_data, tuple) and len(validation_data) == 2:
                        x_val, y_val = validation_data
                        x_val_enc, x_val_dec, y_val_prep = self.prepare_data(x_val, y_val)
                        val_loss, val_predictions = self.evaluate([x_val_enc, x_val_dec], 
                                                                y_val_prep, batch_size)
                    else:
                        raise ValueError("validation_data must be a tuple of (x_val, y_val)")

                    history['val_loss'].append(val_loss)
                    epoch_logs['val_loss'] = val_loss

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

    def predict(self, inp: np.ndarray, max_length: int = 50, beam_size: int = 5,
                alpha: float = 0.6, min_length: int = 2, temperature: float = 0.7) -> np.ndarray:
        enc_output = self.encode(inp, training=False)

        input_length = np.sum(inp[0] != self.PAD_IDX)
        adaptive_max_length = min(max_length, input_length * 2)

        beam_sequences = [[(np.array([[self.SOS_IDX]]), 0.0)]]

        for i in range(adaptive_max_length - 1):
            all_candidates = []

            for sequences in beam_sequences[-1]:
                seq, score = sequences

                if seq[0, -1] == self.EOS_IDX:
                    if len(seq[0]) >= min_length:
                        all_candidates.append((seq, score))
                    continue

                if len(seq[0]) > input_length * 1.5:
                    score -= (len(seq[0]) - input_length * 1.5) * 0.1

                dec_output = self.decode(
                    seq,
                    enc_output,
                    training=False,
                    look_ahead_mask=self.create_look_ahead_mask(seq.shape[1]),
                    padding_mask=self.create_padding_mask(inp)
                )

                logits = self.output_layer.forward_pass(dec_output)[:, -1, :]

                invalid_tokens = np.array(
                    [self.PAD_IDX, self.SOS_IDX, self.UNK_IDX])

                logits[:, invalid_tokens] = -np.inf

                log_probs = log_softmax(logits[0] / temperature)

                top_k = min(beam_size * 2, self.tgt_vocab_size)
                top_indices = np.argpartition(log_probs, -top_k)[-top_k:]

                valid_tokens = np.ones(len(top_indices), dtype=bool)
                valid_tokens[top_indices == self.SOS_IDX] = False
                if len(seq[0]) < min_length:
                    valid_tokens[top_indices == self.EOS_IDX] = False

                for idx, is_valid in zip(top_indices[valid_tokens], valid_tokens[valid_tokens]):
                    candidate_score = score + log_probs[idx]
                    length_penalty = ((5 + len(seq[0]) + 1) / 6) ** alpha
                    candidate_score = candidate_score / length_penalty

                    candidate_seq = np.concatenate([seq, [[idx]]], axis=1)
                    all_candidates.append((candidate_seq, candidate_score))

            if not all_candidates:
                break

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beam_sequences.append(ordered[:beam_size])

            if all(seq[0, -1] == self.EOS_IDX for seq, _ in beam_sequences[-1]):
                break

        best_seq = max(beam_sequences[-1], key=lambda x: x[1])[0]

        if best_seq[0, -1] == self.EOS_IDX:
            return best_seq[:, 1:-1]
        return best_seq[:, 1:]

    def evaluate(self, x_test: list[np.ndarray], y_test: np.ndarray, batch_size: int = 32) -> tuple[float, np.ndarray]:
        if isinstance(x_test, list) and len(x_test) == 2:
            encoder_input, decoder_input = x_test
        else:
            raise ValueError(
                "x_test must be a list of [encoder_input, decoder_input]")

        decoder_target = y_test

        total_loss = 0
        if batch_size is None:
            batch_size = len(encoder_input)
        num_batches = int(np.ceil(len(encoder_input) / batch_size))
        predictions_list = []

        for i in range(0, len(encoder_input), batch_size):
            enc_batch = encoder_input[i:i + batch_size]
            dec_batch = decoder_input[i:i + batch_size]
            target_batch = decoder_target[i:i + batch_size]

            predictions = self.forward_pass(
                (enc_batch, dec_batch), training=False)
            batch_loss = self.loss_function(target_batch, predictions)

            total_loss += batch_loss
            predictions_list.append(predictions)

        avg_loss = total_loss / num_batches
        all_predictions = np.vstack(predictions_list)

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
                json.dump(self.src_embedding.get_config(), f, indent=4)

        if self.tgt_embedding is not None:
            tgt_emb_file = f"{base}_tgt_embedding{ext}"
            config['tgt_embedding_file'] = tgt_emb_file
            with open(tgt_emb_file, 'w') as f:
                json.dump(self.tgt_embedding.get_config(), f, indent=4)

        config['encoder_layers_files'] = []
        for i, layer in enumerate(self.encoder_layers):
            encoder_file = f"{base}_encoder_layer_{i}{ext}"
            config['encoder_layers_files'].append(encoder_file)
            with open(encoder_file, 'w') as f:
                json.dump(layer.get_config(), f, indent=4)

        config['decoder_layers_files'] = []
        for i, layer in enumerate(self.decoder_layers):
            decoder_file = f"{base}_decoder_layer_{i}{ext}"
            config['decoder_layers_files'].append(decoder_file)
            with open(decoder_file, 'w') as f:
                json.dump(layer.get_config(), f, indent=4)

        if self.output_layer is not None:
            output_file = f"{base}_output_layer{ext}"
            config['output_layer_file'] = output_file
            with open(output_file, 'w') as f:
                json.dump(self.output_layer.get_config(), f, indent=4)

        if self.optimizer is not None:
            optimizer_file = f"{base}_optimizer{ext}"
            config['optimizer_file'] = optimizer_file
            with open(optimizer_file, 'w') as f:
                json.dump(self.optimizer.get_config(), f, indent=4)

        if self.loss_function is not None:
            loss_file = f"{base}_loss{ext}"
            config['loss_file'] = loss_file
            with open(loss_file, 'w') as f:
                json.dump(self.loss_function.get_config(), f, indent=4)

        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

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

    @property
    def image_dimensions(self) -> tuple[int, int]:
        if self._image_height is None or self._image_width is None:
            if self.generator is None:
                raise ValueError(
                    "The image dimensions are not defined and the generator is not compiled.")
            return self._infer_dimensions_from_generator()
        return self._image_height, self._image_width

    def _infer_dimensions_from_generator(self) -> tuple[int, int]:
        last_dense = None
        for layer in self.generator.layers:
            if isinstance(layer, Dense):
                last_dense = layer

        if last_dense is None:
            raise ValueError(
                "Generator must contain at least one Dense layer.")

        n_pixels = last_dense.units

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

        if self._image_height is None or self._image_width is None:
            self._image_height, self._image_width = self._infer_dimensions_from_generator()
            if verbose:
                print(
                    f"Inferred image dimensions: {self._image_height}x{self._image_width}")

        if self.n_classes is not None:
            noise_size = self.latent_dim + self.n_classes
            if generator.layers[0].input_dim != noise_size:
                raise ValueError(
                    f"Generator input dimension ({generator.layers[0].input_dim}) "
                    f"does not match expected size (latent_dim + n_classes = {noise_size})"
                )

        last_dense = None
        for layer in generator.layers:
            if isinstance(layer, Dense):
                last_dense = layer

        if last_dense is None:
            raise ValueError(
                "The generator must contain at least one Dense layer.")

        expected_size = self._image_height * self._image_width
        if last_dense.units != expected_size:
            raise ValueError(
                f"The generator must produce images of size {expected_size} "
                f"({self._image_height}x{self._image_width}), "
                f"but the last Dense layer produces {last_dense.units} units."
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
            
        self.last_activation = self.generator.layers[-1].activation_function.get_config()['name'].lower()

    def forward_pass(self, latent_vectors: np.ndarray, training: bool = True) -> np.ndarray:
        if self.generator is None:
            raise ValueError("Model must be compiled before forward pass")

        return self.generator.forward_pass(latent_vectors, training)

    def backward_pass(self, error: np.ndarray):
        if self.generator is None:
            raise ValueError("Model must be compiled before backward pass")

        self.generator.backward_pass(error)

    def _generate_latent_points(self, n_samples: int, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        
        if self.n_classes is not None:
            if labels is not None:
                if labels.ndim == 1:
                    one_hot_labels = np.zeros((len(labels), self.n_classes))
                    one_hot_labels[np.arange(len(labels)), labels] = 1
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
            if hasattr(layer, 'weights'):
                layer.weights = self.spectral_norm(layer.weights)

    def _gradient_penalty(self, real_samples: np.ndarray, fake_samples: np.ndarray) -> float:
        if not self.use_gradient_penalty:
            return 0.0

        rng = np.random.default_rng(self.random_state)
        batch_size = real_samples.shape[0]
        alpha = rng.uniform(0, 1, (batch_size, 1))

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        disc_interpolated = self.discriminator.forward_pass(interpolated)
        gradients = self.discriminator.backward_pass(
            np.ones_like(disc_interpolated),
            compute_only=True
        )

        gradients_norm = np.sqrt(np.sum(np.square(gradients)))
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
        rng = np.random.default_rng(self.random_state)

        d_loss_total = 0
        for _ in range(n_critic):
            idx = rng.choice(len(real_samples), batch_size, replace=False)
            real_batch = real_samples[idx]
            batch_labels = labels[idx] if labels is not None else None

            latent_points, gen_labels = self._generate_latent_points(batch_size, batch_labels)
            fake_batch = self.generator.forward_pass(latent_points, training=False)

            combined_batch = np.concatenate([real_batch, fake_batch])
            if self.n_classes is not None and batch_labels is not None:
                disc_labels = np.concatenate([batch_labels, gen_labels])
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
        self.generator.backward_pass(d_grad, gan=True)

        return d_loss_avg, g_loss

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
                    random_state=random_state
                )
                validation_data = (x_val, y_val)
            else:
                x_train, x_val = train_test_split(
                    x_train, test_size=validation_split, random_state=random_state
                )
                validation_data = (x_val, None)

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        if metrics is not None:
            metrics = [Metric(m) for m in metrics]
            for metric in metrics:
                history[f'discriminator_{metric.name}'] = []
                history[f'generator_{metric.name}'] = []
                if validation_data is not None:
                    history[f'val_discriminator_{metric.name}'] = []
                    history[f'val_generator_{metric.name}'] = []

        if plot_generated:
            if fixed_noise is None:
                n_rows = self.n_classes if self.n_classes is not None else visualization_grid[0]
                n_cols = visualization_grid[1]
                n_samples = n_rows * n_cols
                    
                rng = np.random.default_rng(self.random_state)
                fixed_noise = rng.normal(0, 1, (n_samples, self.latent_dim))
                
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

        try:
            for epoch in range(epochs):
                epoch_logs = {'model': self}
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)

                start_time = time.time()
                if y_train.any():
                    x_train_shuffled, y_train_shuffled = shuffle(
                        x_train, y_train,
                        random_state=random_state if random_state is not None else self.random_state
                    )
                else:
                    x_train_shuffled = shuffle(
                        x_train,
                        random_state=random_state if random_state is not None else self.random_state
                    )
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
                        if y_train.any():
                            y_batch = y_train_shuffled[j:j + batch_size]
                        else:
                            y_batch = None
                        
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
                            latent_points, _  = self._generate_latent_points(len(x_batch), y_batch)
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
                                    f'{metrics_str[:-3]} - '
                                    f'{time.time() - start_time:.2f}s'
                                )
                            )

                    d_error /= num_batches
                    g_error /= num_batches
                    for k in metric_values:
                        metric_values[k] /= num_batches

                else:
                    d_error, g_error = self.train_on_batch(x_train, y_train, len(x_train), n_critic)

                    if metrics is not None:
                        latent_points, _ = self._generate_latent_points(
                            len(x_train) if n_gen_samples is None else n_gen_samples, y_train)
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
                        x_val, y_val = validation_data
                    else:
                        x_val = validation_data
                        y_val = None
                        
                    x_val = np.array(x_val)
                    
                    val_d_loss, val_g_loss = self.evaluate(x_val, y_val, batch_size)
                    
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
        n_rows, n_cols = grid_size
        n_samples = n_rows * n_cols
        
        if noise.shape[0] != n_samples:
            raise ValueError(f"The number of noise samples ({noise.shape[0]}) must match the grid size ({n_samples})")
        
        if self.n_classes is not None:
            latent_points, _ = self._generate_latent_points(n_samples, 
                np.repeat(np.arange(self.n_classes), n_cols))
            
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
            if labels.ndim == 1:
                one_hot_labels = np.zeros((len(labels), self.n_classes))
                one_hot_labels[np.arange(len(labels)), labels] = 1
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
               discriminator_real_input = np.concatenate([x_batch, batch_labels], axis=1)
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

       return avg_d_loss, avg_g_loss

    def save(self, filename: str):
        base, ext = os.path.splitext(filename)
        
        model_state = {
            'type': 'GAN',
            'latent_dim': self.latent_dim,
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state,
            'generator': self.generator.save(f"{base}_generator{ext}") if self.generator else None,
            'discriminator': self.discriminator.save(f"{base}_discriminator{ext}") if self.discriminator else None,
            'generator_optimizer': self.generator_optimizer.get_config() if self.generator_optimizer else None,
            'discriminator_optimizer': self.discriminator_optimizer.get_config() if self.discriminator_optimizer else None,
            'generator_loss': self.generator_loss.get_config() if self.generator_loss else None,
            'discriminator_loss': self.discriminator_loss.get_config() if self.discriminator_loss else None
        }

        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4)

    @classmethod
    def load(cls, filename: str) -> 'GAN':
        base, ext = os.path.splitext(filename)

        with open(filename, 'r') as f:
            model_state = json.load(f)

        model = cls(
            latent_dim=model_state['latent_dim'],
            gradient_clip_threshold=model_state['gradient_clip_threshold'],
            enable_padding=model_state['enable_padding'],
            padding_size=model_state['padding_size'],
            random_state=model_state['random_state']
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
            'generator': [layer.weights.copy() if hasattr(layer, 'weights') else None
                          for layer in self.generator.layers],
            'biases': [layer.bias.copy() if hasattr(layer, 'bias') else None
                       for layer in self.generator.layers],
            'epoch': epoch
        }
        return weights

    def load_weights(self, weights):
        for layer, w, b in zip(self.generator.layers, weights['generator'], weights['biases']):
            if hasattr(layer, 'weights'):
                layer.weights = w.copy()
            if hasattr(layer, 'bias'):
                layer.bias = b.copy()
