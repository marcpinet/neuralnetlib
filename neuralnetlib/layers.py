import time

import numpy as np

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.preprocessing import im2col, col2im


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
        if config['name'] == 'Input':
            return Input.from_config(config)
        elif config['name'] == 'Dense':
            return Dense.from_config(config)
        elif config['name'] == 'Activation':
            return Activation.from_config(config)
        elif config['name'] == 'Conv2D':
            return Conv2D.from_config(config)
        elif config['name'] == 'MaxPooling2D':
            return MaxPooling2D.from_config(config)
        elif config['name'] == 'Flatten':
            return Flatten.from_config(config)
        elif config['name'] == 'Dropout':
            return Dropout.from_config(config)
        else:
            raise ValueError(f'Invalid layer name: {config["name"]}')


class Input(Layer):
    def __init__(self, input_shape: tuple | int):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = input_shape

    def __str__(self):
        return f'Input(input_shape={self.input_shape})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'input_shape': self.input_shape
        }

    @staticmethod
    def from_config(config: dict):
        return Input(config['input_shape'])


class Dense(Layer):
    def __init__(self, units: int, weights_init: str = "default", bias_init: str = "default", random_state: int = None):
        self.units = units

        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state

    def initialize_weights(self, input_size: int):
        self.rng = np.random.default_rng(self.random_state if self.random_state is not None else int(time.time_ns()))
        if self.weights_init == "xavier":
            stddev = np.sqrt(2 / (input_size + self.units))
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))
        elif self.weights_init == "he":
            stddev = np.sqrt(2 / input_size)
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(0, 0.01, (input_size, self.units))
        elif self.weights_init == "lecun":
            stddev = np.sqrt(1 / input_size)
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))
        else:
            raise ValueError("Invalid weights_init value. Possible values are 'xavier', 'he', 'lecun' and 'default'.")

        if self.bias_init == "default":
            self.bias = np.zeros((1, self.units))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, self.units))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, self.units))
        elif self.bias_init == "small":
            self.bias = np.full((1, self.units), 0.01)
        else:
            raise ValueError("Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Dense(units={self.units})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            self.initialize_weights(input_data.shape[1])

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
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'units': self.units,
            'weights_init': self.weights_init,
            'bias_init': self.bias_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Dense(config['units'], config['weights_init'], config['bias_init'], config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
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


class Dropout(Layer):
    def __init__(self, rate: float):
        self.rate = rate
        self.mask = None

    def __str__(self):
        return f'Dropout(rate={self.rate})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        else:
            return input_data

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error * self.mask

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'rate': self.rate
        }

    @staticmethod
    def from_config(config: dict):
        return Dropout(config['rate'])


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int | tuple, stride: int | tuple = 1, padding: str = 'valid', weights_init: str = "default", bias_init: str = "default", random_state: int = None):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding

        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state

    def initialize_weights(self, input_shape: tuple):
        in_channels, _, _ = input_shape

        self.rng = np.random.default_rng(self.random_state if self.random_state is not None else int(time.time_ns()))
        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (np.prod(self.kernel_size) * in_channels)), (self.filters, in_channels, *self.kernel_size))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * np.prod(self.kernel_size))), (self.filters, in_channels, *self.kernel_size))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(0, 0.01, (self.filters, in_channels, *self.kernel_size))
        else:
            raise ValueError("Invalid weights_init value. Possible values are 'xavier', 'he', and 'default'.")

        if self.bias_init == "default":
            self.bias = np.zeros((1, self.filters))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, self.filters))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, self.filters))
        elif self.bias_init == "small":
            self.bias = np.full((1, self.filters), 0.01)
        else:
            raise ValueError("Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Conv2D(num_filters={self.filters}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            self.initialize_weights(input_data.shape[1:])

        self.input = input_data
        output = self._convolve(self.input, self.weights, self.bias, self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error, self.d_weights, self.d_bias = self._convolve_backward(output_error, self.input, self.weights, self.stride, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'weights_init': self.weights_init,
            'bias_init': self.bias_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Conv2D(config['filters'], config['kernel_size'], config['stride'], config['padding'],
                       config['weights_init'], config['bias_init'], config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
        return layer

    @staticmethod
    def _convolve(input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray, stride: tuple,
                  padding: str) -> np.ndarray:
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weights.shape

        assert in_channels == _

        if padding == 'same':
            pad_height = ((in_height - 1) * stride[0] + kernel_height - in_height) // 2
            pad_width = ((in_width - 1) * stride[1] + kernel_width - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        out_height = (in_height + 2 * pad_height - kernel_height) // stride[0] + 1
        out_width = (in_width + 2 * pad_width - kernel_width) // stride[1] + 1

        col = im2col(input_data, kernel_height, kernel_width, stride, (pad_height, pad_width))
        col_W = weights.reshape(out_channels, -1).T

        output = np.dot(col, col_W) + bias
        output = output.reshape(batch_size, out_height, out_width, -1).transpose(0, 3, 1, 2)

        return output

    @staticmethod
    def _convolve_backward(output_error: np.ndarray, input_data: np.ndarray, weights: np.ndarray, stride: tuple,
                           padding: str) -> tuple:
        batch_size, in_channels, in_height, in_width = input_data.shape
        _, out_channels, out_height, out_width = output_error.shape
        _, _, kernel_height, kernel_width = weights.shape

        if padding == 'same':
            pad_height = ((in_height - 1) * stride[0] + kernel_height - in_height) // 2
            pad_width = ((in_width - 1) * stride[1] + kernel_width - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        col = im2col(input_data, kernel_height, kernel_width, stride, (pad_height, pad_width))
        col_W = weights.reshape(out_channels, -1).T

        d_output = output_error.transpose(0, 2, 3, 1).reshape(batch_size * out_height * out_width, -1)
        d_bias = np.sum(d_output, axis=0)
        d_weights = np.dot(col.T, d_output)
        d_weights = d_weights.transpose(1, 0).reshape(weights.shape)

        d_col = np.dot(d_output, col_W.T)
        d_input = col2im(d_col, input_data.shape, kernel_height, kernel_width, stride, (pad_height, pad_width))

        return d_input, d_weights, d_bias


class MaxPooling2D(Layer):
    def __init__(self, pool_size: tuple | int, stride: tuple = None, padding: str = 'valid'):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        self.stride = stride if stride is not None else self.pool_size
        self.padding = padding

    def __str__(self):
        return f'MaxPooling2D(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        output = self._pool(self.input, self.pool_size, self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(output_error, self.input, self.pool_size, self.stride, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }

    @staticmethod
    def from_config(config: dict):
        return MaxPooling2D(config['pool_size'], config['stride'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: tuple, stride: tuple, padding: str) -> np.ndarray:
        batch_size, channels, in_height, in_width = input_data.shape

        if padding == 'same':
            pad_height = ((in_height - 1) * stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) * stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        out_height = (in_height + 2 * pad_height - pool_size[0]) // stride[0] + 1
        out_width = (in_width + 2 * pad_width - pool_size[1]) // stride[1] + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                              j * stride[1]:j * stride[1] + pool_size[1]]
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: tuple, stride: tuple,
                       padding: str) -> np.ndarray:
        batch_size, channels, in_height, in_width = input_data.shape
        _, _, out_height, out_width = output_error.shape

        if padding == 'same':
            pad_height = ((in_height - 1) * stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) * stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                              j * stride[1]:j * stride[1] + pool_size[1]]
                mask = (input_slice == np.max(input_slice, axis=(2, 3), keepdims=True))
                d_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                j * stride[1]:j * stride[1] + pool_size[1]] += output_error[:, :, i, j][:, :, np.newaxis,
                                                               np.newaxis] * mask

        if padding == 'same':
            d_input = d_input[:, :, pad_height:-pad_height, pad_width:-pad_width]

        return d_input


class Flatten(Layer):
    def __str__(self):
        return 'Flatten'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error.reshape(self.input_shape)

    def get_config(self) -> dict:
        return {'name': self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        return Flatten()
