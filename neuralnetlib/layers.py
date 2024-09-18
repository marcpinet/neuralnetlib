import time

import numpy as np
from collections import Counter

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.preprocessing import im2col_2d, col2im_2d, im2col_1d, col2im_1d


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
        elif config['name'] == 'AveragePooling2D':
            return AveragePooling2D.from_config(config)
        elif config['name'] == 'Flatten':
            return Flatten.from_config(config)
        elif config['name'] == 'Dropout':
            return Dropout.from_config(config)
        elif config['name'] == 'Conv1D':
            return Conv1D.from_config(config)
        elif config['name'] == 'MaxPooling1D':
            return MaxPooling1D.from_config(config)
        elif config['name'] == 'AveragePooling1D':
            return AveragePooling1D.from_config(config)
        elif config['name'] == 'Embedding':
            return Embedding.from_config(config)
        elif config['name'] == 'BatchNormalization':
            return BatchNormalization.from_config(config)
        elif config['name'] == 'Permute':
            return Permute.from_config(config)
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
    def __init__(self, units: int, weights_init: str = "default", bias_init: str = "default", random_state: int = None,
                 **kwargs):
        self.units = units

        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_weights(self, input_size: int):
        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
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
            raise ValueError(
                "Invalid weights_init value. Possible values are 'xavier', 'he', 'lecun' and 'default'.")

        if self.bias_init == "default":
            self.bias = np.zeros((1, self.units))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, self.units))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, self.units))
        elif self.bias_init == "small":
            self.bias = np.full((1, self.units), 0.01)
        else:
            raise ValueError(
                "Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Dense(units={self.units})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 2, f"Dense input must be 2D (batch_size, features), got {input_data.shape}"
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
        layer = Dense(config['units'], config['weights_init'],
                      config['bias_init'], config['random_state'])
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
        activation_function = ActivationFunction.from_config(
            config['activation_function'])
        return Activation(activation_function)

    @staticmethod
    def from_name(name: str) -> "Activation":
        name = name.lower().replace("_", "")
        for subclass in ActivationFunction.__subclasses__():
            if subclass.__name__.lower() == name:
                return Activation(subclass())
        raise ValueError(f"No activation function found for the name: {name}")


class Dropout(Layer):
    def __init__(self, rate: float, seed: int = None, **kwargs):
        self.rate = rate
        self.mask = None
        self.seed = seed

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f'Dropout(rate={self.rate})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            rng = np.random.default_rng(self.seed)
            self.mask = rng.binomial(
                1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        else:
            return input_data

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error * self.mask

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'rate': self.rate,
            'seed': self.seed
        }

    @staticmethod
    def from_config(config: dict):
        return Dropout(config['rate'], config['seed'])


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int | tuple, stride: int | tuple = 1, padding: str = 'valid',
                 weights_init: str = "default", bias_init: str = "default", random_state: int = None, **kwargs):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding

        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_weights(self, input_shape: tuple):
        in_channels, _, _ = input_shape

        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (np.prod(self.kernel_size) * in_channels)),
                                           (self.filters, in_channels, *self.kernel_size))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * np.prod(self.kernel_size))),
                                           (self.filters, in_channels, *self.kernel_size))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(
                0, 0.01, (self.filters, in_channels, *self.kernel_size))
        else:
            raise ValueError(
                "Invalid weights_init value. Possible values are 'xavier', 'he', and 'default'.")

        if self.bias_init == "default":
            self.bias = np.zeros((1, self.filters))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, self.filters))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, self.filters))
        elif self.bias_init == "small":
            self.bias = np.full((1, self.filters), 0.01)
        else:
            raise ValueError(
                "Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Conv2D(num_filters={self.filters}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 4, f"Conv2D input must be 4D (batch_size, channels, height, width), got {input_data.shape}"
            self.initialize_weights(input_data.shape[1:])

        self.input = input_data
        output = self._convolve(self.input, self.weights,
                                self.bias, self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error, self.d_weights, self.d_bias = self._convolve_backward(output_error, self.input, self.weights,
                                                                           self.stride, self.padding)
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
            pad_height = ((in_height - 1) *
                          stride[0] + kernel_height - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + kernel_width - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        out_height = (in_height + 2 * pad_height -
                      kernel_height) // stride[0] + 1
        out_width = (in_width + 2 * pad_width - kernel_width) // stride[1] + 1

        col = im2col_2d(input_data, kernel_height, kernel_width,
                        stride, (pad_height, pad_width))
        col_W = weights.reshape(out_channels, -1).T

        output = np.dot(col, col_W) + bias
        output = output.reshape(batch_size, out_height,
                                out_width, -1).transpose(0, 3, 1, 2)

        return output

    @staticmethod
    def _convolve_backward(output_error: np.ndarray, input_data: np.ndarray, weights: np.ndarray, stride: tuple,
                           padding: str) -> tuple:
        batch_size, in_channels, in_height, in_width = input_data.shape
        _, out_channels, out_height, out_width = output_error.shape
        _, _, kernel_height, kernel_width = weights.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          stride[0] + kernel_height - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + kernel_width - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        col = im2col_2d(input_data, kernel_height, kernel_width,
                        stride, (pad_height, pad_width))
        col_W = weights.reshape(out_channels, -1).T

        d_output = output_error.transpose(0, 2, 3, 1).reshape(
            batch_size * out_height * out_width, -1)
        d_bias = np.sum(d_output, axis=0)
        d_weights = np.dot(col.T, d_output)
        d_weights = d_weights.transpose(1, 0).reshape(weights.shape)

        d_col = np.dot(d_output, col_W.T)
        d_input = col2im_2d(d_col, input_data.shape, kernel_height,
                            kernel_width, stride, (pad_height, pad_width))

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
        assert len(
            input_data.shape) == 4, f"MaxPooling2D input must be 4D (batch_size, channels, height, width), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.stride, self.padding)
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
            pad_height = ((in_height - 1) *
                          stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        out_height = (in_height + 2 * pad_height -
                      pool_size[0]) // stride[0] + 1
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
            pad_height = ((in_height - 1) *
                          stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                                           j * stride[1]:j * stride[1] + pool_size[1]]
                mask = (input_slice == np.max(
                    input_slice, axis=(2, 3), keepdims=True))
                d_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                        j * stride[1]:j * stride[1] + pool_size[1]] += output_error[:, :, i, j][:, :, np.newaxis,
                                                                                                np.newaxis] * mask

        if padding == 'same':
            d_input = d_input[:, :, pad_height:-
                              pad_height, pad_width:-pad_width]

        return d_input


class Flatten(Layer):
    def __str__(self):
        return 'Flatten'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) >= 2, f"Flatten input must be at least 2D, got {input_data.shape}"
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error.reshape(self.input_shape)

    def get_config(self) -> dict:
        return {'name': self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        return Flatten()


class Conv1D(Layer):
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, padding: str = 'valid',
                 weights_init: str = "default", bias_init: str = "default", random_state: int = None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_weights(self, input_shape: tuple):
        in_channels = input_shape[0]

        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (self.kernel_size * in_channels)),
                                           (self.filters, in_channels, self.kernel_size))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * self.kernel_size)),
                                           (self.filters, in_channels, self.kernel_size))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(
                0, 0.01, (self.filters, in_channels, self.kernel_size))
        else:
            raise ValueError(
                "Invalid weights_init value. Possible values are 'xavier', 'he', and 'default'.")

        if self.bias_init == "default":
            self.bias = np.zeros((1, self.filters))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, self.filters))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, self.filters))
        elif self.bias_init == "small":
            self.bias = np.full((1, self.filters), 0.01)
        else:
            raise ValueError(
                "Invalid bias_init value. Possible values are 'normal', 'uniform', 'small' and 'default'.")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def __str__(self):
        return f'Conv1D(num_filters={self.filters}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 3, f"Conv1D input must be 3D (batch_size, steps, features), got {input_data.shape}"
            self.initialize_weights(input_data.shape[1:])

        self.input = input_data
        output = self._convolve(self.input, self.weights,
                                self.bias, self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error, self.d_weights, self.d_bias = self._convolve_backward(output_error, self.input, self.weights,
                                                                           self.stride, self.padding)
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
        layer = Conv1D(config['filters'], config['kernel_size'], config['stride'], config['padding'],
                       config['weights_init'], config['bias_init'], config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
        return layer

    @staticmethod
    def _convolve(input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray, stride: int,
                  padding: str) -> np.ndarray:
        batch_size, in_channels, in_length = input_data.shape
        out_channels, _, kernel_length = weights.shape

        assert in_channels == _

        if padding == 'same':
            pad_length = ((in_length - 1) * stride +
                          kernel_length - in_length) // 2
        else:
            pad_length = 0

        out_length = (in_length + 2 * pad_length - kernel_length) // stride + 1

        col = im2col_1d(input_data, kernel_length, stride, pad_length)
        col_W = weights.reshape(out_channels, -1).T

        output = np.dot(col, col_W) + bias
        output = output.reshape(batch_size, out_length, -1).transpose(0, 2, 1)

        return output

    @staticmethod
    def _convolve_backward(output_error: np.ndarray, input_data: np.ndarray, weights: np.ndarray, stride: int,
                           padding: str) -> tuple:
        batch_size, in_channels, in_length = input_data.shape
        _, out_channels, out_length = output_error.shape
        _, _, kernel_length = weights.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * stride +
                          kernel_length - in_length) // 2
        else:
            pad_length = 0

        col = im2col_1d(input_data, kernel_length, stride, pad_length)
        col_W = weights.reshape(out_channels, -1).T

        d_output = output_error.transpose(
            0, 2, 1).reshape(batch_size * out_length, -1)
        d_bias = np.sum(d_output, axis=0)
        d_weights = np.dot(col.T, d_output)
        d_weights = d_weights.transpose(1, 0).reshape(weights.shape)

        d_col = np.dot(d_output, col_W.T)
        d_input = col2im_1d(d_col, input_data.shape,
                            kernel_length, stride, pad_length)

        return d_input, d_weights, d_bias


class MaxPooling1D(Layer):
    def __init__(self, pool_size: int, stride: int = None, padding: str = 'valid'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def __str__(self):
        return f'MaxPooling1D(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 3, f"MaxPooling1D input must be 3D (batch_size, steps, features), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.stride, self.padding)
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
        return MaxPooling1D(config['pool_size'], config['stride'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: int, stride: int, padding: str) -> np.ndarray:
        batch_size, channels, in_length = input_data.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * stride +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (0, 0), (pad_length, pad_length)), mode='constant')

        out_length = (in_length + 2 * pad_length - pool_size) // stride + 1

        output = np.zeros((batch_size, channels, out_length))

        for i in range(out_length):
            input_slice = padded_input[:, :, i * stride:i * stride + pool_size]
            output[:, :, i] = np.max(input_slice, axis=2)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: int, stride: int,
                       padding: str) -> np.ndarray:
        batch_size, channels, in_length = input_data.shape
        _, _, out_length = output_error.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * stride +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (0, 0), (pad_length, pad_length)), mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_length):
            input_slice = padded_input[:, :, i * stride:i * stride + pool_size]
            mask = (input_slice == np.max(input_slice, axis=2, keepdims=True))
            d_input[:, :, i * stride:i * stride +
                    pool_size] += output_error[:, :, i][:, :, np.newaxis] * mask

        if padding == 'same':
            d_input = d_input[:, :, pad_length:-pad_length]

        return d_input


class Embedding(Layer):
    def __init__(self, input_dim: int, output_dim: int, input_length: int = None, weights_init: str = "default",
                 random_state: int = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.weights = None
        self.weights_init = weights_init
        self.random_state = random_state

    def initialize_weights(self):
        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (self.input_dim + self.output_dim)),
                                           (self.input_dim, self.output_dim))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(
                2 / self.input_dim), (self.input_dim, self.output_dim))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(
                0, 0.01, (self.input_dim, self.output_dim))
        else:
            raise ValueError(
                "Invalid weights_init value. Possible values are 'xavier', 'he', and 'default'.")

    def __str__(self):
        return f'Embedding(input_dim={self.input_dim}, output_dim={self.output_dim}, input_length={self.input_length})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 2, f"Embedding input must be 2D (batch_size, sequence_length), got {input_data.shape}"
            self.initialize_weights()

        self.input = input_data
        output = self.weights[input_data]
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = np.zeros(
            (self.input.shape[0], self.input.shape[1], self.input_dim))
        output_error = output_error.reshape(
            output_error.shape[0], output_error.shape[1], -1)
        for i, index in enumerate(self.input):
            input_error[i, np.arange(index.shape[0]), index] = np.sum(
                output_error[i], axis=1)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_length': self.input_length,
            'weights_init': self.weights_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Embedding(config['input_dim'], config['output_dim'], config['input_length'], config['weights_init'],
                          config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
        return layer


class BatchNormalization(Layer):
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-8, **kwargs):
        self.gamma = None
        self.beta = None
        self.d_gamma = None
        self.d_beta = None
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_weights(self, input_shape: tuple):
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.ones(input_shape)

    def __str__(self):
        return f'BatchNormalization(momentum={self.momentum}, epsilon={self.epsilon})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if self.gamma is None:
            self.initialize_weights(input_data.shape[1:])

        if training:
            mean = np.mean(input_data, axis=0)
            var = np.var(input_data, axis=0)
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.input_centered = input_data - mean
        self.input_normalized = self.input_centered / \
            np.sqrt(var + self.epsilon)
        return self.gamma * self.input_normalized + self.beta

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        N = output_error.shape[0]
        self.d_gamma = np.sum(output_error * self.input_normalized, axis=0)
        self.d_beta = np.sum(output_error, axis=0)

        d_input_normalized = output_error * self.gamma
        d_var = np.sum(d_input_normalized * self.input_centered, axis=0) * -0.5 * (
            self.input_centered / (self.input_centered.var(axis=0) + self.epsilon) ** 1.5)
        d_mean = np.sum(d_input_normalized, axis=0) * -1 / np.sqrt(
            self.input_centered.var(axis=0) + self.epsilon) - 2 * d_var * np.mean(self.input_centered, axis=0)
        d_input = d_input_normalized / np.sqrt(
            self.input_centered.var(axis=0) + self.epsilon) + 2 * d_var * self.input_centered / N + d_mean / N
        return d_input

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'gamma': self.gamma.tolist() if self.gamma is not None else None,
            'beta': self.beta.tolist() if self.beta is not None else None,
            'momentum': self.momentum,
            'epsilon': self.epsilon
        }

    @staticmethod
    def from_config(config: dict):
        layer = BatchNormalization(config['momentum'], config['epsilon'])
        if config['gamma'] is not None:
            layer.gamma = np.array(config['gamma'])
            layer.beta = np.array(config['beta'])
        return layer


class AveragePooling2D(Layer):
    def __init__(self, pool_size: tuple | int, stride: tuple = None, padding: str = 'valid'):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        self.stride = stride if stride is not None else self.pool_size
        self.padding = padding

    def __str__(self):
        return f'AveragePooling2D(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 4, f"AveragePooling2D input must be 4D (batch_size, channels, height, width), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.stride, self.padding)
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
        return AveragePooling2D(config['pool_size'], config['stride'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: tuple, stride: tuple, padding: str) -> np.ndarray:
        batch_size, channels, in_height, in_width = input_data.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        out_height = (in_height + 2 * pad_height -
                      pool_size[0]) // stride[0] + 1
        out_width = (in_width + 2 * pad_width - pool_size[1]) // stride[1] + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                                           j * stride[1]:j * stride[1] + pool_size[1]]
                output[:, :, i, j] = np.mean(input_slice, axis=(2, 3))

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: tuple, stride: tuple,
                       padding: str) -> np.ndarray:
        batch_size, channels, in_height, in_width = input_data.shape
        _, _, out_height, out_width = output_error.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          stride[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         stride[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_height):
            for j in range(out_width):
                d_input[:, :, i * stride[0]:i * stride[0] + pool_size[0],
                        j * stride[1]:j * stride[1] + pool_size[1]] += output_error[:, :, i, j][:, :, np.newaxis,
                                                                                                np.newaxis] / np.prod(pool_size)

        if padding == 'same':
            d_input = d_input[:, :, pad_height:-
                              pad_height, pad_width:-pad_width]

        return d_input


class AveragePooling1D(Layer):
    def __init__(self, pool_size: int, stride: int = None, padding: str = 'valid'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def __str__(self):
        return f'AveragePooling1D(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 3, f"AveragePooling1D input must be 3D (batch_size, steps, features), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.stride, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.stride, self.padding)
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
        return AveragePooling1D(config['pool_size'], config['stride'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: int, stride: int, padding: str) -> np.ndarray:
        batch_size, steps, features = input_data.shape

        if padding == 'same':
            pad_steps = ((steps - 1) * stride + pool_size - steps) // 2
        else:
            pad_steps = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_steps, pad_steps), (0, 0)), mode='constant')

        out_steps = (steps + 2 * pad_steps - pool_size) // stride + 1

        output = np.zeros((batch_size, out_steps, features))

        for i in range(out_steps):
            input_slice = padded_input[:, i * stride:i * stride + pool_size, :]
            output[:, i, :] = np.mean(input_slice, axis=1)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: int, stride: int,
                       padding: str) -> np.ndarray:
        batch_size, steps, features = input_data.shape
        _, out_steps, _ = output_error.shape

        if padding == 'same':
            pad_steps = ((steps - 1) * stride + pool_size - steps) // 2
        else:
            pad_steps = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_steps, pad_steps), (0, 0)), mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_steps):
            d_input[:, i * stride:i * stride + pool_size,
                    :] += output_error[:, i, :][:, np.newaxis, :] / pool_size

        if padding == 'same':
            d_input = d_input[:, pad_steps:-pad_steps, :]

        return d_input


class Permute(Layer):
    def __init__(self, dims):
        self.dims = dims

    def __str__(self):
        return f'Permute(dims={self.dims})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        permutation = [0] + [dim - 1 for dim in self.dims]
        self.output = np.transpose(self.input, permutation)
        return self.output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = np.transpose(output_error, np.argsort(
            [0] + [dim - 1 for dim in self.dims]))
        return input_error

    def get_config(self) -> dict:
        config = {'name': self.__class__.__name__, 'dims': self.dims}
        config.update({key: getattr(self, key)
                       for key in self.__dict__ if key not in ['dims']})
        return config

    @staticmethod
    def from_config(config: dict):
        return Permute(config['dims'],
                       **{key: value for key, value in config.items() if key != 'name' and key != 'dims'})


class TextVectorization(Layer):
    def __init__(self, max_tokens=None, output_mode='int', output_sequence_length=None):
        super().__init__()
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        self.vocabulary = None
        self.word_index = None

    def __str__(self):
        return f'TextVectorization(max_tokens={self.max_tokens}, output_mode={self.output_mode}, output_sequence_length={self.output_sequence_length})'

    def adapt(self, data):
        if len(data.shape) == 2:
            data = data.flatten()

        tokens = [word for text in data for word in text.lower().split()]

        token_counts = Counter(tokens)

        sorted_tokens = sorted(token_counts.items(),
                               key=lambda x: x[1], reverse=True)

        if self.max_tokens:
            # -1 to reserve index 0 for padding
            sorted_tokens = sorted_tokens[:self.max_tokens - 1]

        self.vocabulary = [''] + [token for token, _ in sorted_tokens]
        self.word_index = {word: i for i, word in enumerate(self.vocabulary)}

    def forward_pass(self, input_data):
        if isinstance(input_data[0], str):
            vectorized = [[self.word_index.get(
                word, 0) for word in text.lower().split()] for text in input_data]
        else:
            vectorized = [[self.word_index.get(
                word, 0) for word in sequence] for sequence in input_data]

        if self.output_sequence_length:
            vectorized = [seq[:self.output_sequence_length] + [0] *
                          max(0, self.output_sequence_length - len(seq)) for seq in vectorized]

        if self.output_mode == 'int':
            return np.array(vectorized)
        elif self.output_mode == 'binary':
            return np.array([[1 if token > 0 else 0 for token in seq] for seq in vectorized])
        elif self.output_mode == 'count':
            return np.array([[seq.count(token) for token in range(1, len(self.vocabulary))] for seq in vectorized])
        elif self.output_mode == 'tfidf':
            tf = np.array([[seq.count(token) for token in range(
                1, len(self.vocabulary))] for seq in vectorized])
            idf = np.log(len(vectorized) / (1 + np.array([[(token in seq) for seq in vectorized].count(
                True) for token in range(1, len(self.vocabulary))])))
            return tf * idf
        else:
            raise ValueError("Invalid output_mode. Use 'int' or 'binary'.")

    def backward_pass(self, output_error):
        return output_error

    def get_config(self):
        return {
            'name': self.__class__.__name__,
            'max_tokens': self.max_tokens,
            'output_mode': self.output_mode,
            'output_sequence_length': self.output_sequence_length,
            'vocabulary': self.vocabulary
        }

    @staticmethod
    def from_config(config):
        layer = TextVectorization(
            config['max_tokens'], config['output_mode'], config['output_sequence_length'])
        layer.vocabulary = config['vocabulary']
        layer.word_index = {word: i for i, word in enumerate(layer.vocabulary)}
        return layer


class Reshape(Layer):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.input_shape = None

    def __str__(self):
        return f'Reshape(target_shape={self.target_shape})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return np.reshape(input_data, (input_data.shape[0],) + self.target_shape)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return np.reshape(output_error, self.input_shape)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'target_shape': self.target_shape
        }

    @staticmethod
    def from_config(config: dict):
        return Reshape(config['target_shape'])


# --------------------------------------------------------------------------------------------------------------


compatibility_dict = {
    Input: [Dense, Conv2D, Conv1D, Embedding, Permute, TextVectorization, Reshape],
    Dense: [Dense, Activation, Dropout, BatchNormalization, Permute, Reshape],
    Activation: [Dense, Conv2D, Conv1D, MaxPooling2D, AveragePooling2D, MaxPooling1D, AveragePooling1D, Flatten,
                 Dropout, Permute, Reshape],
    Conv2D: [Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, BatchNormalization, Permute, Reshape],
    MaxPooling2D: [Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Permute, Reshape],
    AveragePooling2D: [Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Permute, Reshape],
    Conv1D: [Conv1D, MaxPooling1D, AveragePooling1D, Activation, Dropout, Flatten, BatchNormalization, Permute, Reshape],
    MaxPooling1D: [Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Permute, Reshape],
    AveragePooling1D: [Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Permute, Reshape],
    Flatten: [Dense, Dropout, Permute, Reshape],
    Dropout: [Dense, Conv2D, Conv1D, Activation, Permute, Reshape],
    Embedding: [Conv1D, Flatten, Dense, Permute, Reshape],
    BatchNormalization: [Dense, Conv2D, Conv1D, Activation, Permute, Reshape],
    Permute: [Dense, Conv2D, Conv1D, Activation,
              Dropout, Flatten, BatchNormalization, Permute, Reshape],
    TextVectorization: [Embedding, Dense, Conv1D, Reshape],
    Reshape: [Dense, Conv2D, Conv1D, Activation, Dropout, Flatten, BatchNormalization, Permute, Reshape,
              TextVectorization, Embedding, Input, MaxPooling2D, AveragePooling2D, MaxPooling1D, AveragePooling1D]
}
