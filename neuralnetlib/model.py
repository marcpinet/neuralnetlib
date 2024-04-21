import json
import time

import numpy as np

from neuralnetlib.layers import Layer, Input, Activation, Dense, Flatten, Conv2D, Dropout
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy
from neuralnetlib.optimizers import Optimizer
from neuralnetlib.utils import shuffle, progress_bar
from neuralnetlib.metrics import accuracy_score


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
        if self.layers and len(self.layers) != 0 and not isinstance(self.layers[-1], Input) and isinstance(layer,
                                                                                                           Dense):
            prev_layer = [l for l in self.layers if isinstance(l, (Input, Dense, Conv2D, Flatten))][-1]
            if isinstance(prev_layer, Flatten):
                prev_layer = [l for l in self.layers if isinstance(l, (Dense, Conv2D))][-1]
            if hasattr(prev_layer, 'output_size') and prev_layer.output_size != layer.input_size:
                raise ValueError(
                    f'Layer input size {layer.input_size} does not match previous layer output size {prev_layer.output_size}.')
        elif self.layers and isinstance(layer, Dropout):
            if isinstance(self.layers[-1], Dropout):
                raise ValueError("Cannot add consecutive Dropout layers.")
        self.layers.append(layer)

    def compile(self, loss_function: LossFunction, optimizer: Optimizer, verbose: bool = False):
        self.loss_function = loss_function
        self.optimizer = optimizer
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
                self.optimizer.update(len(self.layers) - 1 - i, layer.weights, layer.d_weights, layer.bias,
                                      layer.d_bias)

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

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = None,
              verbose: bool = True, metrics: list = None, random_state: int = None, validation_data: tuple = None):
        for i in range(epochs):
            start_time = time.time()

            # Shuffling the data to avoid overfitting
            x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=random_state)

            error = 0
            predictions_list = []
            y_true_list = []

            if batch_size is not None:
                num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)
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
                                metric_value = metric(np.vstack(predictions_list), np.vstack(y_true_list))
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
                            metric_value = metric(np.vstack(predictions_list), np.vstack(y_true_list))
                            metrics_str += f'{metric.__name__}: {metric_value:.4f} - '
                    progress_bar(1, 1,
                                 message=f'Epoch {i + 1}/{epochs} - loss: {error:.4f} - {metrics_str[:-3]} - {time.time() - start_time:.2f}s')

            if validation_data is not None:
                x_test, y_test = validation_data
                val_predictions = self.predict(x_test)
                val_accuracy = accuracy_score(val_predictions, y_test)
                if verbose:
                    print(f' - val_accuracy: {val_accuracy:.4f}', end='')

            if verbose:
                print()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        predictions = self.forward_pass(x_test)
        loss = self.loss_function(y_test, predictions)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        model.layers = [Layer.from_config(layer_config) for layer_config in model_state['layers']]
        model.loss_function = LossFunction.from_config(model_state['loss_function'])
        model.optimizer = Optimizer.from_config(model_state['optimizer'])

        return model
