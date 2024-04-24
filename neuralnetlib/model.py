import json
import time

import numpy as np

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.layers import Layer, Input, Activation, Dropout, compatibility_dict
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy
from neuralnetlib.metrics import accuracy_score
from neuralnetlib.optimizers import Optimizer
from neuralnetlib.utils import shuffle, progress_bar


class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.y_true = None
        self.predictions = None

    def __str__(self):
        model_summary = 'Model\n'
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
        for layer in self.layers:
            if isinstance(layer, Dropout):
                X = layer.forward_pass(X, training)
            else:
                X = layer.forward_pass(X)
        return X

    def backward_pass(self, error: np.ndarray):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0 and isinstance(layer, Activation) and type(
                    layer.activation_function).__name__ == "Softmax" and isinstance(self.loss_function,
                                                                                    CategoricalCrossentropy):
                error = self.predictions - self.y_true
            else:
                error = layer.backward_pass(error)

            if hasattr(layer, 'weights'):
                if hasattr(layer, 'd_weights') and hasattr(layer, 'd_bias'):
                    self.optimizer.update(len(self.layers) - 1 - i, layer.weights, layer.d_weights, layer.bias,
                                          layer.d_bias)
                elif hasattr(layer, 'd_weights'):
                    self.optimizer.update(
                        len(self.layers) - 1 - i, layer.weights, layer.d_weights)

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        self.y_true = y_batch
        self.predictions = self.forward_pass(x_batch)
        predictions = self.predictions.copy()
        loss = self.loss_function(y_batch, predictions)
        error = self.loss_function.derivative(y_batch, predictions)

        if error.ndim == 1:
            error = error[:, None]

        self.backward_pass(error)
        return loss

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = None,
            verbose: bool = True, metrics: list = None, random_state: int = None, validation_data: tuple = None,
            callbacks: list = None):
        """
        Fit the model to the training data.

        Args:
            x_train: Training data
            y_train: Training labels
            epochs: Number of epochs to train the model
            batch_size: Number of samples per gradient update
            verbose: Whether to print training progress
            metrics: List of metric functions to evaluate the model
            random_state: Random seed for shuffling the data
            validation_data: Tuple of validation data and labels
            callbacks: List of callback objects (e.g., EarlyStopping)
        """
        x_train = np.array(x_train) if not isinstance(
            x_train, np.ndarray) else x_train
        y_train = np.array(y_train) if not isinstance(
            y_train, np.ndarray) else y_train

        if validation_data is not None:
            x_test, y_test = validation_data
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        for i in range(epochs):
            start_time = time.time()

            # Shuffling the data to avoid overfitting
            x_train_shuffled, y_train_shuffled = shuffle(
                x_train, y_train, random_state=random_state)

            error = 0
            predictions_list = []
            y_true_list = []

            if batch_size is not None:
                num_batches = np.ceil(
                    x_train.shape[0] / batch_size).astype(int)
                for j in range(0, x_train.shape[0], batch_size):
                    x_batch = x_train_shuffled[j:j + batch_size]
                    y_batch = y_train_shuffled[j:j + batch_size]

                    # Reshape if it's a regression (single output neuron)
                    if y_batch.ndim == 1:
                        y_batch = y_batch.reshape(-1, 1)
                    error += self.train_on_batch(x_batch, y_batch)
                    predictions_list.append(self.predictions)
                    y_true_list.append(y_batch)

                    if verbose:
                        metrics_str = ''
                        if metrics is not None:
                            for metric in metrics:
                                metric_value = metric(
                                    np.vstack(predictions_list), np.vstack(y_true_list))
                                metrics_str += f'{metric.__name__}: {metric_value:.4f} - '
                        progress_bar(j / batch_size + 1, num_batches,
                                     message=f'Epoch {i + 1}/{epochs} - loss: {error / (j / batch_size + 1):.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

                error /= num_batches
            else:
                error = self.train_on_batch(x_train, y_train)
                predictions_list.append(self.predictions)
                y_true_list.append(y_train)

                if verbose:
                    metrics_str = ''
                    if metrics is not None:
                        for metric in metrics:
                            metric_value = metric(
                                np.vstack(predictions_list), np.vstack(y_true_list))
                            metrics_str += f'{metric.__name__}: {metric_value:.4f} - '
                    progress_bar(1, 1,
                                 message=f'Epoch {i + 1}/{epochs} - loss: {error:.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

            if validation_data is not None:
                x_test, y_test = validation_data
                val_predictions = self.predict(x_test)
                val_metrics = []
                if metrics is not None:
                    for metric in metrics:
                        # Change extend to append
                        val_metrics.append(metric(val_predictions, y_test))
                if verbose:
                    val_metrics_str = ' - '.join(
                        f'{metric.__name__}: {val_metric:.4f}' for metric, val_metric in zip(metrics, val_metrics))
                    print(f' - {val_metrics_str}', end='')

            if callbacks:
                metrics_values = {}
                if metrics is not None:
                    for metric in metrics:
                        metrics_values[metric.__name__] = metric(
                            np.vstack(predictions_list), np.vstack(y_true_list))
                        
                callback_monitor_metrics = set(cb.monitor[0].__name__ for cb in callbacks if hasattr(cb, 'monitor') and cb.monitor is not None)
                missing_metrics = callback_monitor_metrics.difference(metrics_values.keys())
                if missing_metrics:
                    raise ValueError(f"The following metrics weren't (and must be) included in the fit() method: {', '.join(missing_metrics)}")

                for callback in callbacks:
                    if callback.stop_training:
                        break
                    if callback.on_epoch_end(self, error, metrics_values):
                        break

                if any(callback.stop_training for callback in callbacks):
                    break

            if verbose:
                print()

        if verbose:
            print()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        predictions = self.forward_pass(x_test)
        loss = self.loss_function(y_test, predictions)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        return self.forward_pass(X, training=False)

    def save(self, filename: str):
        model_state = {
            'layers': [layer.get_config() for layer in self.layers],
            'loss_function': self.loss_function.get_config(),
            'optimizer': self.optimizer.get_config(),
        }
        with open(filename, 'w') as f:
            json.dump(model_state, f, indent=4)

    @staticmethod
    def load(filename: str) -> 'Model':
        with open(filename, 'r') as f:
            model_state = json.load(f)

        model = Model()
        model.layers = [Layer.from_config(layer_config)
                        for layer_config in model_state['layers']]
        model.loss_function = LossFunction.from_config(
            model_state['loss_function'])
        model.optimizer = Optimizer.from_config(model_state['optimizer'])

        return model
