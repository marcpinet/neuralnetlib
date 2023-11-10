import time

import numpy as np

from neuralnetlib.activations import ActivationFunction


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input_data: np.ndarray):
        raise NotImplementedError

    def backward_pass(self, output_error: np.ndarray):
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_config(config: dict) -> 'Layer':
        if config['name'] == 'Dense':
            return Dense.from_config(config)
        elif config['name'] == 'Activation':
            return Activation.from_config(config)


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, weights_init: str = "default", bias_init: str = "default",
                 random_state: int = None):
        self.input_size = input_size
        self.output_size = output_size

        self.rng = np.random.default_rng(random_state if random_state is not None else int(time.time_ns()))
        if weights_init == "xavier":
            stddev = np.sqrt(2 / (input_size + output_size))
            self.weights = self.rng.normal(0, stddev, (input_size, output_size))
        elif weights_init == "he":
            stddev = np.sqrt(2 / input_size)
            self.weights = self.rng.normal(0, stddev, (input_size, output_size))
        elif weights_init == "default":
            self.weights = self.rng.normal(0, 0.01, (input_size, output_size))
        elif weights_init == "lecun":
            stddev = np.sqrt(1 / input_size)
            self.weights = self.rng.normal(0, stddev, (input_size, output_size))
        else:
            raise ValueError("Invalid weights_init value. Possible values are 'xavier', 'he', 'lecun' and 'default'.")

        if bias_init == "default":
            self.bias = np.zeros((1, output_size))
        elif bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, output_size))
        elif bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, output_size))
        elif bias_init == "small":
            self.bias = np.full((1, output_size), 0.01)
        else:
            raise ValueError("Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Dense(num_input={self.weights.shape[0]}, num_output={self.weights.shape[1]})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = np.dot(output_error, self.weights.T)
        self.d_weights = np.dot(self.input.T, output_error)
        self.d_bias = np.sum(output_error, axis=0, keepdims=True)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist()
        }

    @staticmethod
    def from_config(config: dict):
        weights = np.array(config['weights'])
        bias = np.array(config['bias'])
        layer = Dense(weights.shape[0], weights.shape[1])
        layer.weights = weights
        layer.bias = bias
        return layer


class Activation(Layer):
    def __init__(self, activation_function: ActivationFunction):
        super().__init__()
        self.activation_function = activation_function

    def __str__(self):
        name = type(self.activation_function).__name__
        return f'Activation({name})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error * self.activation_function.derivative(self.input)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'activation_function': {
                'name': self.activation_function.__class__.__name__,
                'config': self.activation_function.get_config() if hasattr(self.activation_function,
                                                                           'get_config') else {}
            }
        }

    @staticmethod
    def from_config(config: dict):
        activation_function = ActivationFunction.from_config(config['activation_function'])
        return Activation(activation_function)
