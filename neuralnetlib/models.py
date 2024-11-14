import inspect
import json
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.callbacks import EarlyStopping
from neuralnetlib.layers import compatibility_dict, Layer, Input, Activation, Dropout, TextVectorization, LSTM, GRU, \
    Bidirectional, Embedding, Attention, Dense
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
from neuralnetlib.metrics import Metric
from neuralnetlib.optimizers import Optimizer
from neuralnetlib.preprocessing import PCA
from neuralnetlib.utils import shuffle, progress_bar, is_interactive, is_display_available, History


class BaseModel(ABC):
    def __init__(self, temperature: float = 1.0,
                 gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None):
        
        self.temperature = temperature
        self.gradient_clip_threshold = gradient_clip_threshold
        self.enable_padding = enable_padding
        self.padding_size = padding_size
        self.random_state = random_state if random_state is not None else time.time_ns()

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
    def compile(self, loss_function, optimizer, verbose: bool = False):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, apply_temperature: bool = True) -> np.ndarray:
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

    def set_temperature(self, temperature: float):
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        self.temperature = temperature


class Sequential(BaseModel):
    def __init__(self, temperature: float = 1.0,
                 gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None):
        super().__init__(temperature, gradient_clip_threshold, 
                        enable_padding, padding_size, random_state)
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.y_true = None
        self.predictions = None

    def __str__(self) -> str:
        model_summary = f'Sequential(temperature={self.temperature}, gradient_clip_threshold={self.gradient_clip_threshold}, enable_padding={self.enable_padding}, padding_size={self.padding_size}, random_state={self.random_state})\n'
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
            previous_layer = self.layers[-1]
            if type(layer) not in compatibility_dict[type(previous_layer)]:
                raise ValueError(
                    f"{type(layer).__name__} layer cannot follow {type(previous_layer).__name__} layer.")
            if isinstance(previous_layer, Attention) and isinstance(layer, Dense):
                layer.return_sequences = False

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

    def compile(self, loss_function: LossFunction | str, optimizer: Optimizer | str, verbose: bool = False):
        self.loss_function = loss_function if isinstance(loss_function, LossFunction) else LossFunction.from_name(
            loss_function)
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else Optimizer.from_name(optimizer)
        if verbose:
            print(str(self))

    def forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if self.enable_padding:
            original_shape = X.shape
            padded_shape = ((original_shape[0] + self.padding_size - 1) //
                            self.padding_size * self.padding_size,) + original_shape[1:]

            if padded_shape != original_shape:
                padded_X = np.zeros(padded_shape, dtype=X.dtype)
                padded_X[:original_shape[0]] = X
                X = padded_X

        for layer in self.layers:
            if isinstance(layer, (Dropout, LSTM, Bidirectional, GRU)):
                X = layer.forward_pass(X, training)
            else:
                X = layer.forward_pass(X)

        if self.enable_padding and padded_shape != original_shape:
            X = X[:original_shape[0]]

        return X

    def backward_pass(self, error: np.ndarray):
        def clip_gradients(gradient: np.ndarray) -> np.ndarray:
            if self.gradient_clip_threshold > 0:
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > self.gradient_clip_threshold:
                    return gradient * (self.gradient_clip_threshold / grad_norm)
            return gradient

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0 and isinstance(layer, Activation):
                if (type(layer.activation_function).__name__ == "Softmax" and
                        isinstance(self.loss_function, CategoricalCrossentropy)):
                    error = self.predictions - self.y_true

                elif (type(layer.activation_function).__name__ == "Sigmoid" and
                      isinstance(self.loss_function, BinaryCrossentropy)):
                    error = (self.predictions - self.y_true) / (self.predictions *
                                                                (1 - self.predictions) + 1e-15)

                elif isinstance(self.loss_function, SparseCategoricalCrossentropy):
                    y_true_one_hot = np.zeros_like(self.predictions)
                    y_true_one_hot[np.arange(len(self.y_true)), self.y_true] = 1
                    error = self.predictions - y_true_one_hot
            else:
                error = clip_gradients(error)
                error = layer.backward_pass(error)

            layer_idx = len(self.layers) - 1 - i

            if isinstance(layer, LSTM):
                cell = layer.cell
                for grad_pair in [(cell.dWf, cell.dbf), (cell.dWi, cell.dbi),
                                  (cell.dWc, cell.dbc), (cell.dWo, cell.dbo)]:
                    weight_grad, bias_grad = grad_pair
                    clipped_weight_grad = clip_gradients(weight_grad)
                    clipped_bias_grad = clip_gradients(bias_grad)

                self.optimizer.update(layer_idx, cell.Wf, clipped_weight_grad,
                                      cell.bf, clipped_bias_grad)
                self.optimizer.update(layer_idx, cell.Wi, clip_gradients(cell.dWi),
                                      cell.bi, clip_gradients(cell.dbi))
                self.optimizer.update(layer_idx, cell.Wc, clip_gradients(cell.dWc),
                                      cell.bc, clip_gradients(cell.dbc))
                self.optimizer.update(layer_idx, cell.Wo, clip_gradients(cell.dWo),
                                      cell.bo, clip_gradients(cell.dbo))

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
                    self.optimizer.update(layer_idx, layer.weights, clipped_weights_grad)

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

        history = History({
            'loss': [],
            'val_loss': []
        })

        if plot_decision_boundary and not is_interactive() and not is_display_available():
            raise ValueError("Cannot display the plot. Please run the script in an environment with a display.")

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        # Set the random_state for every layer that has a random_state attribute
        for layer in self.layers:
            if hasattr(layer, 'random_state'):
                layer.random_state = random_state if random_state is not None else self.random_state

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU)) for layer in self.layers)
        has_embedding = any(isinstance(layer, Embedding) for layer in self.layers)

        if has_lstm_or_gru and not has_embedding:
            if len(x_train.shape) != 3:
                raise ValueError(
                    "Input data must be 3D (batch_size, time_steps, features) for LSTM/GRU layers without Embedding")
        elif has_embedding:
            if len(x_train.shape) != 2:
                raise ValueError("Input data must be 2D (batch_size, sequence_length) when using Embedding layer")

        if validation_data is not None:
            x_test, y_test = validation_data
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        if metrics is not None:
            metrics: list[Metric] = [Metric(m) for m in metrics]
            for metric in metrics:
                history[metric.name] = []
                history[f'val_{metric.name}'] = []

        for layer in self.layers:
            if isinstance(layer, TextVectorization):
                layer.adapt(x_train)
                break

        if callbacks is None:
            callbacks = []

        for callback in callbacks:
            callback.on_train_begin()

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            start_time = time.time()
            x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train,
                                                         random_state=random_state if random_state is not None else self.random_state)
            error = 0
            predictions_list = []
            y_true_list = []

            if batch_size is not None:
                num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)
                for j in range(0, x_train.shape[0], batch_size):
                    x_batch = x_train_shuffled[j:j + batch_size]
                    y_batch = y_train_shuffled[j:j + batch_size]
                    if y_batch.ndim == 1:
                        y_batch = y_batch.reshape(-1, 1)
                    error += self.train_on_batch(x_batch, y_batch)
                    predictions_list.append(self.predictions)
                    y_true_list.append(y_batch)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(np.vstack(predictions_list), np.vstack(y_true_list))
                                metrics_str += f'{metric.name}: {metric_value:.4f} - '
                        progress_bar(j / batch_size + 1, num_batches,
                                     message=f'Epoch {epoch + 1}/{epochs} - loss: {error / (j / batch_size + 1):.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

                error /= num_batches
            else:
                error = self.train_on_batch(x_train, y_train)
                predictions_list.append(self.predictions)
                y_true_list.append(y_train)

                if verbose:
                    metrics_str = ''
                    if metrics is not None:
                        for metric in metrics:
                            metric_value = metric(np.vstack(predictions_list), np.vstack(y_true_list))
                            history[metric.name].append(metric_value)
                            metrics_str += f'{metric.name}: {metric_value:.4f} - '
                    progress_bar(1, 1,
                                 message=f'Epoch {epoch + 1}/{epochs} - loss: {error:.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

            history['loss'].append(error)

            logs = {'loss': error}
            if metrics is not None:
                for metric in metrics:
                    metric_value = metric(np.vstack(predictions_list), np.vstack(y_true_list))
                    logs[metric.name] = metric_value

            if validation_data is not None:
                x_test, y_test = validation_data
                val_loss, val_predictions = self.evaluate(x_test, y_test, batch_size)
                history['val_loss'].append(val_loss)
                logs['val_loss'] = val_loss

                if metrics is not None:
                    val_metrics = []
                    for metric in metrics:
                        val_metric = metric(val_predictions, y_test)
                        history[f'val_{metric.name}'].append(val_metric)
                        logs[f'val_{metric.name}'] = val_metric
                        val_metrics.append(val_metric)
                    if verbose:
                        val_metrics_str = ' - '.join(
                            f'val_{metric.name}: {val_metric:.4f}'
                            for metric, val_metric in zip(metrics, val_metrics)
                        )
                        print(f' - {val_metrics_str}', end='')

                val_predictions = None

            stop_training = False
            for callback in callbacks:
                if isinstance(callback, EarlyStopping):
                    if callback.on_epoch_end(epoch, {**logs, 'model': self}):
                        stop_training = True
                        break
                else:
                    callback.on_epoch_end(epoch, logs)

            if verbose:
                print()

            if plot_decision_boundary:
                self.__update_plot(epoch, x_train, y_train,
                                   random_state if random_state is not None else self.random_state)
                plt.pause(0.1)

            if stop_training:
                break

        if plot_decision_boundary:
            plt.show(block=True)

        for callback in callbacks:
            callback.on_train_end()

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

    def predict(self, X: np.ndarray, apply_temperature: bool = True) -> np.ndarray:
        X = np.array(X)
        predictions = self.forward_pass(X, training=False)

        if apply_temperature and not np.isclose(self.temperature, 1.0, rtol=1e-09, atol=1e-09):
            if isinstance(predictions, np.ndarray):
                predictions = np.clip(predictions, 1e-7, 1.0)
                log_preds = np.log(predictions)
                scaled_log_preds = log_preds / self.temperature
                predictions = np.exp(scaled_log_preds)
                predictions /= np.sum(predictions, axis=-1, keepdims=True)

        return predictions

    def generate_sequence(self,
                          sequence_start: np.ndarray,
                          max_length: int,
                          stop_token: int | None = None,
                          min_length: int | None = None) -> np.ndarray:

        current_sequence = sequence_start.copy()
        batch_size = current_sequence.shape[0]

        for _ in range(max_length - sequence_start.shape[1]):
            predictions = self.predict(current_sequence,
                                       apply_temperature=False)  # cuz we already apply temperature in this method

            if predictions.ndim == 3:
                next_token_probs = predictions[:, -1, :]
            else:
                next_token_probs = predictions

            if not np.isclose(self.temperature, 1.0, rtol=1e-09, atol=1e-09):
                next_token_probs = np.clip(next_token_probs, 1e-7, 1.0)
                log_probs = np.log(next_token_probs)
                scaled_log_probs = log_probs / self.temperature
                next_token_probs = np.exp(scaled_log_probs)
                next_token_probs /= np.sum(next_token_probs, axis=-1, keepdims=True)

            if min_length is not None and current_sequence.shape[1] < min_length:
                if stop_token is not None:
                    next_token_probs[:, stop_token] = 0
                    next_token_probs /= np.sum(next_token_probs, axis=-1, keepdims=True)

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

            current_sequence = np.hstack([current_sequence, next_tokens.reshape(-1, 1)])

            self.random_state += 1

        return current_sequence

    def set_temperature(self, temperature: float):
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        self.temperature = temperature

    def save(self, filename: str):
        model_state = {
            'type': 'Sequential',
            'layers': [],
            'temperature': self.temperature,
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state
        }
        
        for layer in self.layers:
            model_state['layers'].append(layer.get_config())

        if self.loss_function:
            model_state['loss_function'] = self.loss_function.get_config()
        if self.optimizer:
            model_state['optimizer'] = self.optimizer.get_config()

        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4)

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
            model.loss_function = LossFunction.from_config(model_state['loss_function'])
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

        scatter = ax.scatter(x_train_2d[:, 0], x_train_2d[:, 1], c=y_train_encoded, cmap='viridis', alpha=0.7)

        labels = np.unique(y_train_encoded)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)),
                       label=f'Class {label}', markersize=8) for label in labels]
        ax.legend(handles=handles, title='Classes')

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(pca.inverse_transform(grid_points))
        if Z.shape[1] > 1:  # Multiclass classification
            Z = np.argmax(Z, axis=1).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu, levels=np.arange(Z.max() + 1))
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
                 temperature: float = 1.0,
                 gradient_clip_threshold: float = 5.0,
                 enable_padding: bool = False,
                 padding_size: int = 32,
                 random_state: int | None = None,
                 skip_connections: bool = False,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.0,
                 variational: bool = False):
        super().__init__(temperature, gradient_clip_threshold, 
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

    def _calculate_kl_divergence(self):
        if not self.variational:
            return 0.0
        kl_loss = -0.5 * np.sum(1 + self.latent_log_var - np.square(self.latent_mean) - np.exp(self.latent_log_var))
        return kl_loss / len(self.latent_mean)

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
                raise ValueError("The first encoder layer must be an Input layer.")
        else:
            previous_layer = self.encoder_layers[-1]
            if type(layer) not in compatibility_dict[type(previous_layer)]:
                raise ValueError(
                    f"{type(layer).__name__} layer cannot follow {type(previous_layer).__name__} layer in encoder.")

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
                raise ValueError(f"Invalid activation function: {activation_attr}")
            self.encoder_layers.append(activation)
    
    def add_decoder_layer(self, layer: Layer):
        if self.decoder_layers:
            previous_layer = self.decoder_layers[-1]
            if type(layer) not in compatibility_dict[type(previous_layer)]:
                raise ValueError(
                    f"{type(layer).__name__} layer cannot follow {type(previous_layer).__name__} layer in decoder.")
        
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
                raise ValueError(f"Invalid activation function: {activation_attr}")
            self.decoder_layers.append(activation)
    
    def compile(self, 
                encoder_loss: LossFunction | str = None,
                decoder_loss: LossFunction | str = None,
                encoder_optimizer: Optimizer | str = None,
                decoder_optimizer: Optimizer | str = None,
                verbose: bool = False):
        
        if encoder_loss is None:
            encoder_loss = decoder_loss
        if decoder_loss is None:
            decoder_loss = encoder_loss
        if encoder_optimizer is None:
            encoder_optimizer = decoder_optimizer
        if decoder_optimizer is None:
            decoder_optimizer = encoder_optimizer
            
        if encoder_loss is None or encoder_optimizer is None:
            raise ValueError("At least one loss and optimizer must be specified")
            
        self.encoder_loss = encoder_loss if isinstance(encoder_loss, LossFunction) else LossFunction.from_name(encoder_loss)
        self.decoder_loss = decoder_loss if isinstance(decoder_loss, LossFunction) else LossFunction.from_name(decoder_loss)
        self.encoder_optimizer = encoder_optimizer if isinstance(encoder_optimizer, Optimizer) else Optimizer.from_name(encoder_optimizer)
        self.decoder_optimizer = decoder_optimizer if isinstance(decoder_optimizer, Optimizer) else Optimizer.from_name(decoder_optimizer)
        
        if verbose:
            print(str(self))
            
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
                    gradient = gradient * (self.gradient_clip_threshold / grad_norm)
                    
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
                self._update_rnn_weights(layer, layer_idx, clip_gradients, self.decoder_optimizer)
            elif hasattr(layer, 'weights'):
                self._update_layer_weights(layer, layer_idx, clip_gradients, self.decoder_optimizer)
        
        # Encoder backward pass
        for i, layer in enumerate(reversed(self.encoder_layers)):
            error = clip_gradients(error)
            error = layer.backward_pass(error)
            
            layer_idx = len(self.encoder_layers) - 1 - i
            
            if isinstance(layer, (LSTM, GRU)):
                self._update_rnn_weights(layer, layer_idx, clip_gradients, self.encoder_optimizer)
            elif hasattr(layer, 'weights'):
                self._update_layer_weights(layer, layer_idx, clip_gradients, self.encoder_optimizer)
                
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
        
        total_loss = reconstruction_loss + regularization_loss + latent_l2 + distribution_penalty + kl_loss
        
        error = self.decoder_loss.derivative(y_batch, self.predictions)
        if error.ndim == 1:
            error = error[:, None]
        elif isinstance(self.decoder_layers[-1], (LSTM, Bidirectional, GRU)) and self.decoder_layers[-1].return_sequences:
            error = error.reshape(error.shape[0], error.shape[1], -1)
                
        self.backward_pass(error)
        return total_loss
    
    def predict(self, X: np.ndarray, output_latent: bool = False, apply_temperature: bool = True) -> np.ndarray:
        X = np.array(X)
        encoded = X
        for layer in self.encoder_layers:
            encoded = layer.forward_pass(encoded)
            
        if output_latent:
            return encoded
            
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer.forward_pass(decoded)
            
        if apply_temperature and not np.isclose(self.temperature, 1.0, rtol=1e-09, atol=1e-09):
            if isinstance(decoded, np.ndarray):
                decoded = np.clip(decoded, 1e-7, 1.0)
                log_preds = np.log(decoded)
                scaled_log_preds = log_preds / self.temperature
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
            model.encoder_loss = LossFunction.from_config(model_state['encoder_loss'])
        if 'decoder_loss' in model_state:
            model.decoder_loss = LossFunction.from_config(model_state['decoder_loss'])
        if 'encoder_optimizer' in model_state:
            model.encoder_optimizer = Optimizer.from_config(model_state['encoder_optimizer'])
        if 'decoder_optimizer' in model_state:
            model.decoder_optimizer = Optimizer.from_config(model_state['decoder_optimizer'])

        return model
        
    def save(self, filename: str):
        model_state = {
            'type': 'Autoencoder',
            'encoder_layers': [],
            'decoder_layers': [],
            'temperature': self.temperature,
            'gradient_clip_threshold': self.gradient_clip_threshold,
            'enable_padding': self.enable_padding,
            'padding_size': self.padding_size,
            'random_state': self.random_state,
            'skip_connections': self.skip_connections,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
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
        model_summary = f'Autoencoder(temperature={self.temperature}, gradient_clip_threshold={self.gradient_clip_threshold}, ' \
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
            callbacks: list = []) -> dict:

        history = History({
            'loss': [],
            'val_loss': []
        })

        x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train

        # Set the random_state for layers
        for layer in self.encoder_layers + self.decoder_layers:
            if hasattr(layer, 'random_state'):
                layer.random_state = random_state if random_state is not None else self.random_state

        has_lstm_or_gru = any(isinstance(layer, (LSTM, Bidirectional, GRU)) 
                            for layer in self.encoder_layers + self.decoder_layers)
        has_embedding = any(isinstance(layer, Embedding) 
                        for layer in self.encoder_layers + self.decoder_layers)

        # Validate input shape for RNN layers
        if has_lstm_or_gru and not has_embedding:
            if len(x_train.shape) != 3:
                raise ValueError(
                    "Input data must be 3D (batch_size, time_steps, features) for LSTM/GRU layers without Embedding")
        elif has_embedding:
            if len(x_train.shape) != 2:
                raise ValueError("Input data must be 2D (batch_size, sequence_length) when using Embedding layer")

        # Handle validation data
        if validation_data is not None:
            x_val, y_val = validation_data if len(validation_data) == 2 else (validation_data[0], validation_data[0])
            x_val = np.array(x_val)
            y_val = np.array(y_val)

        # Initialize metrics
        if metrics is not None:
            metrics: list[Metric] = [Metric(m) for m in metrics]
            for metric in metrics:
                history[metric.name] = []
                history[f'val_{metric.name}'] = []

        # Initialize text vectorization if present
        for layer in self.encoder_layers + self.decoder_layers:
            if isinstance(layer, TextVectorization):
                layer.adapt(x_train)
                break

        # Initialize callbacks
        if callbacks is None:
            callbacks = []

        for callback in callbacks:
            callback.on_train_begin()

        # Training loop
        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            start_time = time.time()
            y_train = np.zeros_like(x_train)
            x_train_shuffled, _ = shuffle(x_train, y_train, random_state=random_state if random_state is not None else self.random_state)
            
            error = 0
            predictions_list = []
            inputs_list = []

            if batch_size is not None:
                num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)
                
                for j in range(0, x_train.shape[0], batch_size):
                    x_batch = x_train_shuffled[j:j + batch_size]
                    
                    error += self.train_on_batch(x_batch)
                    predictions_list.append(self.predictions)
                    inputs_list.append(x_batch)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(np.vstack(predictions_list), np.vstack(inputs_list))
                                metrics_str += f'{metric.name}: {metric_value:.4f} - '
                        progress_bar(j / batch_size + 1, num_batches,
                                    message=f'Epoch {epoch + 1}/{epochs} - loss: {error / (j / batch_size + 1):.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

                error /= num_batches
                
            else:
                error = self.train_on_batch(x_train)
                predictions_list.append(self.predictions)
                inputs_list.append(x_train)

                if verbose:
                    metrics_str = ''
                    if metrics is not None:
                        for metric in metrics:
                            metric_value = metric(np.vstack(predictions_list), np.vstack(inputs_list))
                            history[metric.name].append(metric_value)
                            metrics_str += f'{metric.name}: {metric_value:.4f} - '
                    progress_bar(1, 1,
                                message=f'Epoch {epoch + 1}/{epochs} - loss: {error:.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

            history['loss'].append(error)

            logs = {'loss': error}
            if metrics is not None:
                for metric in metrics:
                    metric_value = metric(np.vstack(predictions_list), np.vstack(inputs_list))
                    logs[metric.name] = metric_value

            # Validation phase
            if validation_data is not None:
                val_loss, val_predictions = self.evaluate(x_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
                logs['val_loss'] = val_loss

                if metrics is not None:
                    val_metrics = []
                    for metric in metrics:
                        val_metric = metric(val_predictions, x_val)
                        history[f'val_{metric.name}'].append(val_metric)
                        logs[f'val_{metric.name}'] = val_metric
                        val_metrics.append(val_metric)
                    if verbose:
                        val_metrics_str = ' - '.join(
                            f'val_{metric.name}: {val_metric:.4f}'
                            for metric, val_metric in zip(metrics, val_metrics)
                        )
                        print(f' - {val_metrics_str}', end='')

                val_predictions = None

            # Handle callbacks
            stop_training = False
            for callback in callbacks:
                if isinstance(callback, EarlyStopping):
                    if callback.on_epoch_end(epoch, {**logs, 'model': self}):
                        stop_training = True
                        break
                else:
                    callback.on_epoch_end(epoch, logs)

            if verbose:
                print()

            if stop_training:
                break

        for callback in callbacks:
            callback.on_train_end()

        if verbose:
            print()

        return history
