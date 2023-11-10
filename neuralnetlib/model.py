import json

import numpy as np
import time

from neuralnetlib.layers import Layer, Activation, Dense
from neuralnetlib.losses import LossFunction, CategoricalCrossentropy
from neuralnetlib.optimizers import Optimizer


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

    def add(self, layer: Layer):
        if self.layers and isinstance(layer, Dense):
            prev_layer = [l for l in self.layers if isinstance(l, Dense)][-1]
            if hasattr(prev_layer, 'output_size') and prev_layer.output_size != layer.input_size:
                raise ValueError(f'Layer input size {layer.input_size} does not match previous layer output size {prev_layer.output_size}.')
        self.layers.append(layer)

    def __check_layer_compatability(self, layer: Dense) -> bool:
        if len(self.layers) == 0:
            return True
        else:
            return layer.input_size == [l for l in self.layers if isinstance(l, Dense)][-1].output_size

    def compile(self, loss_function: LossFunction, optimizer: Optimizer, verbose: bool = True):
        self.loss_function = loss_function
        self.optimizer = optimizer
        if verbose:
            print(str(self))

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
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
        self.backward_pass(error)
        return loss

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = None,
            verbose: bool = True, metrics: list = None, random_state: int = None):
        rng = np.random.default_rng(random_state if random_state is not None else int(time.time_ns()))
        
        for i in range(epochs):
            # Shuffling the data to avoid overfitting
            indices = rng.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            error = 0
            predictions_list = []
            y_true_list = []

            if batch_size is not None:
                num_batches = np.ceil(x_train.shape[0] / batch_size).astype(int)
                for j in range(0, x_train.shape[0], batch_size):
                    x_batch = x_train_shuffled[j:j + batch_size]
                    y_batch = y_train_shuffled[j:j + batch_size]
                    error += self.train_on_batch(x_batch, y_batch)
                    
                    predictions_list.append(self.predictions)
                    y_true_list.append(y_batch)

                error /= num_batches
            else:
                error = self.train_on_batch(x_train, y_train)
                predictions_list.append(self.predictions)
                y_true_list.append(y_train)

            # Concatenate all predictions and true labels
            all_predictions = np.vstack(predictions_list)
            all_y_true = np.vstack(y_true_list)

            if verbose:
                if metrics is not None:
                    metrics_str = ''
                    for metric in metrics:
                        metric_value = metric(all_predictions, all_y_true)
                        metrics_str += f'{metric.__name__}: {metric_value} - '
                    print(f'Epoch {i + 1}/{epochs} - loss: {error} - {metrics_str[:-3]}')
                else:
                    print(f'Epoch {i + 1}/{epochs} - loss: {error}')


    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        predictions = self.forward_pass(x_test)
        loss = self.loss_function(y_test, predictions)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_pass(X)

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
