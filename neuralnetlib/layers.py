import time
import numpy as np

from collections import Counter

from neuralnetlib.activations import ActivationFunction
from neuralnetlib.preprocessing import im2col_2d, col2im_2d, im2col_1d, col2im_1d, normalize_gradient
from neuralnetlib.regularizers import AdaptiveDropout


EPSILON_SIGMOID = 1e-12


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
        layer_name = config['name']
        try:
            layer_class = globals()[layer_name]
            return layer_class.from_config(config)
        except KeyError:
            raise ValueError(
                f'Invalid layer name: {layer_name}. Make sure the class {layer_name} is defined.')


class Input(Layer):
    def __init__(self, input_shape: tuple | int):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = input_shape

    def __str__(self) -> str:
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
    def __init__(self, units: int, weights_init: str = "glorot_uniform", bias_init: str = "zeros",
                 random_state: int = None, init_scale: float = 1.0, input_dim: int = None,
                 **kwargs):
        super().__init__()
        self.units = units
        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.random_state = random_state
        self.init_scale = init_scale
        self.input_dim = input_dim

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f'Dense(units={self.units})'

    def initialize_weights(self, input_size: int):
        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))

        fan_in = input_size
        fan_out = self.units

        if self.weights_init == "scaled_normal":
            stddev = self.init_scale / np.sqrt(fan_in)
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))
        elif self.weights_init in ["glorot_uniform", "xavier_uniform"]:
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = self.rng.uniform(-limit,
                                            limit, (input_size, self.units))

        elif self.weights_init in ["glorot_normal", "xavier_normal"]:
            stddev = np.sqrt(2 / (fan_in + fan_out))
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))

        elif self.weights_init == "he_uniform":
            limit = np.sqrt(6 / fan_in)
            self.weights = self.rng.uniform(-limit,
                                            limit, (input_size, self.units))

        elif self.weights_init == "he_normal":
            stddev = np.sqrt(2 / fan_in)
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))

        elif self.weights_init == "lecun_uniform":
            limit = np.sqrt(3 / fan_in)
            self.weights = self.rng.uniform(-limit,
                                            limit, (input_size, self.units))

        elif self.weights_init == "lecun_normal":
            stddev = np.sqrt(1 / fan_in)
            self.weights = self.rng.normal(0, stddev, (input_size, self.units))

        elif self.weights_init == "orthogonal":
            nums = self.rng.normal(0, 1, (input_size, self.units))
            q, r = np.linalg.qr(nums)
            q = q * np.sign(np.diag(r))
            self.weights = q

        else:
            raise ValueError(
                "Invalid weights_init value. Possible values are 'scaled_normal', 'glorot_uniform', 'glorot_normal', "
                "'he_uniform', 'he_normal', 'lecun_uniform', 'lecun_normal', 'orthogonal'")

        if self.bias_init == "zeros":
            self.bias = np.zeros((1, self.units))
        elif self.bias_init == "ones":
            self.bias = np.ones((1, self.units))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.05, (1, self.units))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.05, 0.05, (1, self.units))
        else:
            raise ValueError(
                "Invalid bias_init value. Possible values are 'zeros', 'ones', 'normal', 'uniform'")

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        self.input = input_data

        if len(input_data.shape) == 1 and self.input_dim:
            batch_size = input_data.shape[0]
            input_data = input_data.reshape(batch_size, self.input_dim)
            self.input = input_data

        if len(input_data.shape) == 3:
            batch_size, timesteps, features = input_data.shape
            input_reshaped = input_data.reshape(-1, features)

            if self.weights is None:
                self.initialize_weights(features)

            output = np.dot(input_reshaped, self.weights) + self.bias
            return output.reshape(batch_size, timesteps, self.units)

        if self.weights is None:
            self.initialize_weights(input_data.shape[1])

        output = np.dot(input_data, self.weights) + self.bias
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        if len(output_error.shape) == 3:
            batch_size, timesteps, _ = output_error.shape
            output_error_reshaped = output_error.reshape(
                -1, output_error.shape[-1])
            input_reshaped = self.input.reshape(-1, self.input.shape[-1])

            input_error = np.dot(output_error_reshaped, self.weights.T)
            self.d_weights = np.dot(input_reshaped.T, output_error_reshaped)
            self.d_bias = np.sum(output_error_reshaped, axis=0, keepdims=True)

            return input_error.reshape(batch_size, timesteps, -1)

        input_error = np.dot(output_error, self.weights.T)

        if len(self.input.shape) == 1:
            self.input = self.input.reshape(-1, 1)
        if len(output_error.shape) == 1:
            output_error = output_error.reshape(-1, 1)

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
    def __init__(self, activation_function: ActivationFunction | str):
        super().__init__()
        self.activation_function = activation_function if isinstance(
            activation_function, ActivationFunction) else ActivationFunction.from_name(activation_function)

    def __str__(self) -> str:
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
    def __init__(self,
                 rate: float = 0.5,
                 adaptive: bool = False,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 temperature: float = 1.0,
                 random_state: int = None,
                 **kwargs):

        super().__init__()
        self.rate = rate
        self.adaptive = adaptive
        self.random_state = random_state
        self.mask = None

        if adaptive:
            self.dropout_impl = AdaptiveDropout(
                initial_rate=rate,
                min_rate=min_rate,
                max_rate=max_rate,
                temperature=temperature,
                random_state=random_state
            )
        else:
            self.dropout_impl = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        if self.adaptive:
            return f'Dropout(adaptive=True, initial_rate={self.rate})'
        return f'Dropout(rate={self.rate})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return input_data

        if self.adaptive:
            return self.dropout_impl(input_data, training)

        rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
        self.mask = rng.binomial(1, 1 - self.rate,
                                 size=input_data.shape) / (1 - self.rate)
        return input_data * self.mask

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        if self.adaptive:
            return self.dropout_impl.gradient(output_error)
        return output_error * self.mask

    def get_config(self) -> dict:
        config = {
            'name': self.__class__.__name__,
            'rate': self.rate,
            'adaptive': self.adaptive,
            'min_rate': self.dropout_impl.min_rate if self.adaptive else 0.1,
            'max_rate': self.dropout_impl.max_rate if self.adaptive else 0.9,
            'temperature': self.dropout_impl.temperature if self.adaptive else 1.0,
            'random_state': self.random_state
        }
        return config

    @staticmethod
    def from_config(config: dict):
        return Dropout(
            rate=config['rate'],
            adaptive=config['adaptive'],
            min_rate=config['min_rate'],
            max_rate=config['max_rate'],
            temperature=config['temperature'],
            random_state=config['random_state']
        )


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int | tuple, strides: int | tuple = 1, padding: str = 'valid',
                 weights_init: str = "default", bias_init: str = "default", random_state: int = None, **kwargs):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(
            strides, int) else strides
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
        _, _, _, in_channels = input_shape

        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))

        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (np.prod(self.kernel_size) * in_channels)),
                                           (*self.kernel_size, in_channels, self.filters))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * np.prod(self.kernel_size))),
                                           (*self.kernel_size, in_channels, self.filters))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(
                0, 0.01, (*self.kernel_size, in_channels, self.filters))
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

    def __str__(self) -> str:
        return f'Conv2D(num_filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 4, f"Conv2D input must be 4D (batch_size, height, width, channels), got {input_data.shape}"
            self.initialize_weights(input_data.shape)

        self.input = input_data
        output = self._convolve(self.input, self.weights,
                                self.bias, self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error, self.d_weights, self.d_bias = self._convolve_backward(output_error, self.input, self.weights,
                                                                           self.strides, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'weights_init': self.weights_init,
            'bias_init': self.bias_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Conv2D(config['filters'], config['kernel_size'], config['strides'], config['padding'],
                       config['weights_init'], config['bias_init'], config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
        return layer

    @staticmethod
    def _convolve(input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray, strides: tuple,
                  padding: str) -> np.ndarray:
        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_height, kernel_width, _, out_channels = weights.shape

        assert in_channels == weights.shape[2], "Number of input channels must match"

        if padding == 'same':
            out_height = int(np.ceil(float(in_height) / float(strides[0])))
            out_width = int(np.ceil(float(in_width) / float(strides[1])))

            pad_height_total = int(
                max(0, (out_height - 1) * strides[0] + kernel_height - in_height))
            pad_width_total = int(
                max(0, (out_width - 1) * strides[1] + kernel_width - in_width))

            pad_height = pad_height_total // 2
            pad_width = pad_width_total // 2

            out_height = (in_height + 2 * pad_height -
                          kernel_height) // strides[0] + 1
            out_width = (in_width + 2 * pad_width -
                         kernel_width) // strides[1] + 1
        else:
            pad_height, pad_width = 0, 0
            out_height = (in_height - kernel_height) // strides[0] + 1
            out_width = (in_width - kernel_width) // strides[1] + 1

        col = im2col_2d(input_data, kernel_height, kernel_width,
                        strides, (pad_height, pad_width))

        col_W = weights.reshape(-1, out_channels)

        output = np.dot(col, col_W)
        output = output + bias

        expected_elements = batch_size * out_height * out_width * out_channels
        actual_elements = output.size

        if expected_elements != actual_elements:
            raise ValueError(f"Size mismatch: Expected {expected_elements} elements "
                             f"({batch_size}×{out_height}×{out_width}×{out_channels}), "
                             f"but got {actual_elements} elements.")

        output = output.reshape(batch_size, out_height,
                                out_width, out_channels)
        return output

    @staticmethod
    def _convolve_backward(output_error: np.ndarray, input_data: np.ndarray, weights: np.ndarray, strides: tuple,
                           padding: str) -> tuple:
        batch_size, in_height, in_width, in_channels = input_data.shape
        batch_size, out_height, out_width, out_channels = output_error.shape
        kernel_height, kernel_width, _, _ = weights.shape

        if padding == 'same':
            out_height_temp = int(
                np.ceil(float(in_height) / float(strides[0])))
            out_width_temp = int(np.ceil(float(in_width) / float(strides[1])))

            pad_height_total = int(
                max(0, (out_height_temp - 1) * strides[0] + kernel_height - in_height))
            pad_width_total = int(
                max(0, (out_width_temp - 1) * strides[1] + kernel_width - in_width))

            pad_height = pad_height_total // 2
            pad_width = pad_width_total // 2
        else:
            pad_height, pad_width = 0, 0

        col = im2col_2d(input_data, kernel_height, kernel_width,
                        strides, (pad_height, pad_width))

        col_W = weights.reshape(-1, out_channels)

        d_output = output_error.reshape(
            batch_size * out_height * out_width, -1)

        d_bias = np.sum(d_output, axis=0)
        d_weights = np.dot(col.T, d_output)
        d_weights = d_weights.reshape(
            kernel_height, kernel_width, in_channels, out_channels)
        d_col = np.dot(d_output, col_W.T)

        d_input = col2im_2d(d_col, input_data.shape, kernel_height,
                            kernel_width, strides, (pad_height, pad_width))

        return d_input, d_weights, d_bias


class MaxPooling2D(Layer):
    def __init__(self, pool_size: tuple | int, strides: tuple = None, padding: str = 'valid'):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        self.strides = strides if strides is not None else self.pool_size
        self.padding = padding

    def __str__(self) -> str:
        return f'MaxPooling2D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 4, f"MaxPooling2D input must be 4D (batch_size, channels, height, width), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.strides, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        }

    @staticmethod
    def from_config(config: dict):
        return MaxPooling2D(config['pool_size'], config['strides'], config['padding'])

    @staticmethod
    def from_config(config: dict):
        return MaxPooling2D(config['pool_size'], config['strides'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: tuple, strides: tuple, padding: str) -> np.ndarray:
        batch_size, in_height, in_width, channels = input_data.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          strides[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         strides[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data,
                              ((0, 0), (pad_height, pad_height),
                               (pad_width, pad_width), (0, 0)),
                              mode='constant')

        out_height = (in_height + 2 * pad_height -
                      pool_size[0]) // strides[0] + 1
        out_width = (in_width + 2 * pad_width - pool_size[1]) // strides[1] + 1

        output = np.zeros((batch_size, out_height, out_width, channels))

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:,
                                           i * strides[0]:i * strides[0] + pool_size[0],
                                           j * strides[1]:j * strides[1] + pool_size[1],
                                           :]
                output[:, i, j, :] = np.max(
                    np.max(input_slice, axis=1), axis=1)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: tuple, strides: tuple,
                       padding: str) -> np.ndarray:
        batch_size, in_height, in_width, channels = input_data.shape
        _, out_height, out_width, _ = output_error.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          strides[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         strides[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data,
                              ((0, 0), (pad_height, pad_height),
                               (pad_width, pad_width), (0, 0)),
                              mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:,
                                           i * strides[0]:i * strides[0] + pool_size[0],
                                           j * strides[1]:j * strides[1] + pool_size[1],
                                           :]
                mask = (input_slice == np.max(np.max(input_slice, axis=1, keepdims=True),
                                              axis=2, keepdims=True))

                d_input[:,
                        i * strides[0]:i * strides[0] + pool_size[0],
                        j * strides[1]:j * strides[1] + pool_size[1],
                        :] += output_error[:, i:i+1, j:j+1, :] * mask

        if padding == 'same':
            d_input = d_input[:, pad_height:-
                              pad_height, pad_width:-pad_width, :]

        return d_input


class Flatten(Layer):
    def __str__(self) -> str:
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
    def __init__(self, filters: int, kernel_size: int, strides: int = 1, padding: str = 'valid',
                 weights_init: str = "default", bias_init: str = "default", random_state: int = None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
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
        _, _, in_channels = input_shape

        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))

        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (self.kernel_size * in_channels)),
                                           (self.kernel_size, in_channels, self.filters))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * self.kernel_size)),
                                           (self.kernel_size, in_channels, self.filters))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(
                0, 0.01, (self.kernel_size, in_channels, self.filters))
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

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(
                input_data.shape) == 3, f"Conv1D input must be 3D (batch_size, length, channels), got {input_data.shape}"
            self.initialize_weights(input_data.shape)

        self.input = input_data
        output = self._convolve(self.input, self.weights,
                                self.bias, self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error, self.d_weights, self.d_bias = self._convolve_backward(output_error, self.input, self.weights,
                                                                           self.strides, self.padding)
        return input_error

    @staticmethod
    def _convolve(input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray, strides: int,
                  padding: str) -> np.ndarray:
        batch_size, in_length, in_channels = input_data.shape
        kernel_length, _, out_channels = weights.shape

        assert in_channels == weights.shape[1], "Number of input channels must match"

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          kernel_length - in_length) // 2
        else:
            pad_length = 0

        out_length = (in_length + 2 * pad_length -
                      kernel_length) // strides + 1

        col = im2col_1d(input_data, kernel_length, strides, pad_length)

        col_W = weights.reshape(-1, out_channels)

        output = np.dot(col, col_W) + bias

        output = output.reshape(batch_size, out_length, out_channels)

        return output

    @staticmethod
    def _convolve_backward(output_error: np.ndarray, input_data: np.ndarray, weights: np.ndarray, strides: int,
                           padding: str) -> tuple:
        batch_size, in_length, in_channels = input_data.shape
        batch_size, out_length, out_channels = output_error.shape
        kernel_length, _, _ = weights.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          kernel_length - in_length) // 2
        else:
            pad_length = 0

        col = im2col_1d(input_data, kernel_length, strides, pad_length)

        col_W = weights.reshape(-1, out_channels)

        d_output = output_error.reshape(batch_size * out_length, -1)

        d_bias = np.sum(d_output, axis=0)
        d_weights = np.dot(col.T, d_output)

        d_weights = d_weights.reshape(kernel_length, in_channels, out_channels)

        d_col = np.dot(d_output, col_W.T)

        d_input = col2im_1d(d_col, input_data.shape,
                            kernel_length, strides, pad_length)

        return d_input, d_weights, d_bias

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'weights_init': self.weights_init,
            'bias_init': self.bias_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Conv1D(config['filters'], config['kernel_size'], config['strides'], config['padding'],
                       config['weights_init'], config['bias_init'], config['random_state'])
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
        return layer


class MaxPooling1D(Layer):
    def __init__(self, pool_size: int, strides: int = None, padding: str = 'valid'):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding

    def __str__(self) -> str:
        return f'MaxPooling1D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 3, f"MaxPooling1D input must be 3D (batch_size, length, channels), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.strides, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        }

    @staticmethod
    def from_config(config: dict):
        return MaxPooling1D(config['pool_size'], config['strides'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: int, strides: int, padding: str) -> np.ndarray:
        batch_size, in_length, channels = input_data.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_length, pad_length), (0, 0)), mode='constant')

        out_length = (in_length + 2 * pad_length - pool_size) // strides + 1

        output = np.zeros((batch_size, out_length, channels))

        for i in range(out_length):
            input_slice = padded_input[:, i *
                                       strides:i * strides + pool_size, :]
            output[:, i, :] = np.max(input_slice, axis=1)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: int, strides: int,
                       padding: str) -> np.ndarray:
        batch_size, in_length, channels = input_data.shape
        _, out_length, _ = output_error.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_length, pad_length), (0, 0)), mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_length):
            input_slice = padded_input[:, i *
                                       strides:i * strides + pool_size, :]
            mask = (input_slice == np.max(input_slice, axis=1, keepdims=True))
            d_input[:, i * strides:i * strides + pool_size, :] += (
                output_error[:, i, :][:, np.newaxis, :] * mask
            )

        if padding == 'same':
            d_input = d_input[:, pad_length:-pad_length, :]

        return d_input


class AveragePooling2D(Layer):
    def __init__(self, pool_size: tuple | int, strides: tuple = None, padding: str = 'valid'):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        self.strides = strides if strides is not None else self.pool_size
        self.padding = padding

    def __str__(self) -> str:
        return f'AveragePooling2D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 4, f"AveragePooling2D input must be 4D (batch_size, height, width, channels), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.strides, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        }

    @staticmethod
    def from_config(config: dict):
        return AveragePooling2D(config['pool_size'], config['strides'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: tuple, strides: tuple, padding: str) -> np.ndarray:
        batch_size, in_height, in_width, channels = input_data.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          strides[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         strides[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data,
                              ((0, 0), (pad_height, pad_height),
                               (pad_width, pad_width), (0, 0)),
                              mode='constant')

        out_height = (in_height + 2 * pad_height -
                      pool_size[0]) // strides[0] + 1
        out_width = (in_width + 2 * pad_width - pool_size[1]) // strides[1] + 1

        output = np.zeros((batch_size, out_height, out_width, channels))

        for i in range(out_height):
            for j in range(out_width):
                input_slice = padded_input[:,
                                           i * strides[0]:i * strides[0] + pool_size[0],
                                           j * strides[1]:j * strides[1] + pool_size[1],
                                           :]
                output[:, i, j, :] = np.mean(
                    np.mean(input_slice, axis=1), axis=1)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: tuple, strides: tuple,
                       padding: str) -> np.ndarray:
        batch_size, in_height, in_width, channels = input_data.shape
        _, out_height, out_width, _ = output_error.shape

        if padding == 'same':
            pad_height = ((in_height - 1) *
                          strides[0] + pool_size[0] - in_height) // 2
            pad_width = ((in_width - 1) *
                         strides[1] + pool_size[1] - in_width) // 2
        else:
            pad_height, pad_width = 0, 0

        padded_input = np.pad(input_data,
                              ((0, 0), (pad_height, pad_height),
                               (pad_width, pad_width), (0, 0)),
                              mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_height):
            for j in range(out_width):
                d_input[:,
                        i * strides[0]:i * strides[0] + pool_size[0],
                        j * strides[1]:j * strides[1] + pool_size[1],
                        :] += output_error[:, i:i+1, j:j+1, :] / np.prod(pool_size)

        if padding == 'same':
            d_input = d_input[:, pad_height:-
                              pad_height, pad_width:-pad_width, :]

        return d_input


class AveragePooling1D(Layer):
    def __init__(self, pool_size: int, strides: int = None, padding: str = 'valid'):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding

    def __str__(self) -> str:
        return f'AveragePooling1D(pool_size={self.pool_size}, strides={self.strides}, padding={self.padding})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 3, f"AveragePooling1D input must be 3D (batch_size, length, channels), got {input_data.shape}"
        self.input = input_data
        output = self._pool(self.input, self.pool_size,
                            self.strides, self.padding)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_error = self._pool_backward(
            output_error, self.input, self.pool_size, self.strides, self.padding)
        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        }

    @staticmethod
    def from_config(config: dict):
        return AveragePooling1D(config['pool_size'], config['strides'], config['padding'])

    @staticmethod
    def _pool(input_data: np.ndarray, pool_size: int, strides: int, padding: str) -> np.ndarray:
        batch_size, in_length, channels = input_data.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_length, pad_length), (0, 0)), mode='constant')

        out_length = (in_length + 2 * pad_length - pool_size) // strides + 1

        output = np.zeros((batch_size, out_length, channels))

        for i in range(out_length):
            input_slice = padded_input[:, i *
                                       strides:i * strides + pool_size, :]
            output[:, i, :] = np.mean(input_slice, axis=1)

        return output

    @staticmethod
    def _pool_backward(output_error: np.ndarray, input_data: np.ndarray, pool_size: int, strides: int,
                       padding: str) -> np.ndarray:
        batch_size, in_length, channels = input_data.shape
        _, out_length, _ = output_error.shape

        if padding == 'same':
            pad_length = ((in_length - 1) * strides +
                          pool_size - in_length) // 2
        else:
            pad_length = 0

        padded_input = np.pad(
            input_data, ((0, 0), (pad_length, pad_length), (0, 0)), mode='constant')

        d_input = np.zeros_like(padded_input)

        for i in range(out_length):
            d_input[:, i * strides:i * strides + pool_size, :] += (
                output_error[:, i, :][:, np.newaxis, :] / pool_size
            )

        if padding == 'same':
            d_input = d_input[:, pad_length:-pad_length, :]

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
        self.clipped_input = None

    def __str__(self) -> str:
        return f'Embedding(input_dim={self.input_dim}, output_dim={self.output_dim})'

    def initialize_weights(self):
        self.rng = np.random.default_rng(self.random_state)

        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self.weights = self.rng.normal(
            0, scale, (self.input_dim, self.output_dim))

        self.weights[0] = np.zeros(self.output_dim)

        for idx in [1, 2, 3]:  # UNK, SOS, EOS
            special_vector = self.rng.normal(0, scale / 2, self.output_dim)
            self.weights[idx] = special_vector

        epsilon = 1e-8
        norms = np.linalg.norm(self.weights[4:], axis=1, keepdims=True)
        norms = np.maximum(norms, epsilon)
        self.weights[4:] = self.weights[4:] / norms * np.sqrt(self.output_dim)

        self.d_weights = np.zeros_like(self.weights)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            self.initialize_weights()

        input_data = np.clip(input_data, 0, self.input_dim - 1)
        self.clipped_input = input_data.copy()

        output = self.weights[input_data]

        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        batch_size, seq_length, _ = output_error.shape
        grad_weights = np.zeros_like(self.weights)

        seq_length = min(seq_length, self.clipped_input.shape[1])
        input_indices = self.clipped_input[:batch_size, :seq_length]
        flattened_output = output_error[:batch_size, :seq_length]

        mask = (input_indices != 0)

        token_counts = np.bincount(
            input_indices[mask].flatten(),
            minlength=self.input_dim
        )
        token_counts = np.maximum(token_counts, 1)

        for b in range(batch_size):
            grad_batch = flattened_output[b]
            indices_batch = input_indices[b]

            valid_indices = indices_batch != 0
            grad_batch = grad_batch[valid_indices]
            indices_batch = indices_batch[valid_indices]

            grad_batch = grad_batch / token_counts[indices_batch, np.newaxis]
            np.add.at(grad_weights, indices_batch, grad_batch)

        grad_norm = np.linalg.norm(grad_weights[1:])
        if grad_norm > 1.0:
            grad_weights[1:] = grad_weights[1:] / grad_norm

        self.d_weights = grad_weights

        return np.zeros((batch_size, seq_length))

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
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
            layer.bias = np.array(config['bias'])
        return layer


class BatchNormalization(Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-5, **kwargs):
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

    def __str__(self) -> str:
        return f'BatchNormalization(momentum={self.momentum}, epsilon={self.epsilon})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if self.gamma is None or self.running_mean is None or self.running_var is None:
            self.initialize_weights(input_data.shape[1:])

        input_data = np.clip(input_data, -10, 10)

        if training:
            self.batch_mean = np.mean(input_data, axis=0)
            self.batch_var = np.var(input_data, axis=0) + self.epsilon

            self.running_mean = self.momentum * self.running_mean + \
                (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + \
                (1 - self.momentum) * self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        self.input = input_data
        self.std = np.sqrt(var + self.epsilon)
        self.input_centered = input_data - mean
        self.input_normalized = self.input_centered / self.std

        return self.gamma * self.input_normalized + self.beta

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        N = output_error.shape[0]

        self.d_gamma = np.sum(output_error * self.input_normalized, axis=0)
        self.d_beta = np.sum(output_error, axis=0)

        d_normalized = output_error * self.gamma

        d_var = np.sum(d_normalized * self.input_centered * -0.5 *
                       (self.batch_var + self.epsilon) ** (-1.5), axis=0)

        d_mean = np.sum(d_normalized * -1/self.std, axis=0) + \
            d_var * np.mean(-2 * self.input_centered, axis=0)

        d_input = d_normalized / self.std + \
            d_var * 2 * self.input_centered / N + \
            d_mean / N

        return d_input

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'gamma': self.gamma.tolist() if self.gamma is not None else None,
            'beta': self.beta.tolist() if self.beta is not None else None,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'running_mean': self.running_mean.tolist() if self.running_mean is not None else None,
            'running_var': self.running_var.tolist() if self.running_var is not None else None,
            'input_shape': self.gamma.shape if self.gamma is not None else None
        }

    @staticmethod
    def from_config(config: dict):
        layer = BatchNormalization(config['momentum'], config['epsilon'])
        
        if config['gamma'] is not None:
            layer.gamma = np.array(config['gamma'])
            layer.beta = np.array(config['beta'])
            
            layer.running_mean = np.array(config['running_mean'])
            layer.running_var = np.array(config['running_var'])
            
            layer.d_gamma = np.zeros_like(layer.gamma)
            layer.d_beta = np.zeros_like(layer.beta)
        
        return layer


class LayerNormalization(Layer):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.d_gamma = None
        self.d_beta = None

    def initialize_weights(self, input_shape: tuple):
        feature_shape = input_shape[-1:]
        self.gamma = np.ones(feature_shape)
        self.beta = np.zeros(feature_shape)
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            self.initialize_weights(input_data.shape)

        self.input_shape = input_shape = input_data.shape
        self.input = input_data

        if len(input_shape) == 3:
            input_data = input_data.reshape(-1, input_shape[-1])

        self.mean = np.mean(input_data, axis=-1, keepdims=True)
        self.var = np.var(input_data, axis=-1, keepdims=True) + self.epsilon
        self.std = np.sqrt(self.var)

        self.x_centered = input_data - self.mean
        self.x_norm = self.x_centered / self.std

        if len(input_shape) == 3:
            self.x_norm = self.x_norm.reshape(input_shape)
            self.mean = self.mean.reshape(input_shape[:-1] + (1,))
            self.var = self.var.reshape(input_shape[:-1] + (1,))
            self.std = self.std.reshape(input_shape[:-1] + (1,))
            self.x_centered = self.x_centered.reshape(input_shape)

        return self.gamma * self.x_norm + self.beta

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        input_shape = self.input_shape

        if len(input_shape) == 3:
            output_error = output_error.reshape(-1, input_shape[-1])
            x_norm = self.x_norm.reshape(-1, input_shape[-1])
            std = self.std.reshape(-1, 1)
        else:
            x_norm = self.x_norm
            std = self.std

        N = output_error.shape[-1]

        self.d_gamma = np.sum(output_error * x_norm, axis=0, keepdims=True)
        self.d_beta = np.sum(output_error, axis=0, keepdims=True)

        dx_norm = output_error * self.gamma

        dx = (1.0 / std) * (
            dx_norm -
            np.mean(dx_norm, axis=-1, keepdims=True) -
            x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)
        )

        if len(input_shape) == 3:
            dx = dx.reshape(input_shape)

        return dx

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'epsilon': self.epsilon,
            'gamma': self.gamma.tolist() if self.gamma is not None else None,
            'beta': self.beta.tolist() if self.beta is not None else None
        }

    @staticmethod
    def from_config(config: dict):
        layer = LayerNormalization(config['epsilon'])
        if config.get('gamma') is not None:
            layer.gamma = np.array(config['gamma'])
            layer.beta = np.array(config['beta'])
        return layer


class GlobalAveragePooling1D(Layer):
    def __init__(self):
        self.input_shape = None

    def __str__(self) -> str:
        return 'GlobalAveragePooling1D'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 3, f"GlobalAveragePooling1D input must be 3D (batch_size, length, channels), got {input_data.shape}"
        self.input_shape = input_data.shape
        return np.mean(input_data, axis=1)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return np.repeat(output_error[:, np.newaxis, :], self.input_shape[1], axis=1) / self.input_shape[1]

    def get_config(self) -> dict:
        return {'name': self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        return GlobalAveragePooling1D()


class GlobalAveragePooling2D(Layer):
    def __init__(self):
        self.input_shape = None

    def __str__(self) -> str:
        return 'GlobalAveragePooling2D'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        assert len(
            input_data.shape) == 4, f"GlobalAveragePooling2D input must be 4D (batch_size, height, width, channels), got {input_data.shape}"
        self.input_shape = input_data.shape
        return np.mean(input_data, axis=(1, 2))

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.expand_dims(output_error, axis=1), axis=1) * np.ones(
            (1, self.input_shape[1], self.input_shape[2], 1)
        ) / (self.input_shape[1] * self.input_shape[2])

    def get_config(self) -> dict:
        return {'name': self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        return GlobalAveragePooling2D()


class Permute(Layer):
    def __init__(self, dims: tuple):
        self.dims = dims

    def __str__(self) -> str:
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
    def __init__(self, max_tokens: int | None = None, output_mode: str = 'int',
                 output_sequence_length: int | None = None):
        super().__init__()
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        self.vocabulary = None
        self.word_index = None

    def __str__(self) -> str:
        return f'TextVectorization(max_tokens={self.max_tokens}, output_mode={self.output_mode}, output_sequence_length={self.output_sequence_length})'

    def adapt(self, data: np.ndarray):
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

    def forward_pass(self, input_data: np.ndarray | list[str]) -> np.ndarray:
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

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return output_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'max_tokens': self.max_tokens,
            'output_mode': self.output_mode,
            'output_sequence_length': self.output_sequence_length,
            'vocabulary': self.vocabulary,
            'word_index': self.word_index
        }

    @staticmethod
    def from_config(config: dict):
        layer = TextVectorization(
            config['max_tokens'], config['output_mode'], config['output_sequence_length'])
        layer.vocabulary = config['vocabulary']
        layer.word_index = config['word_index']
        return layer


class Reshape(Layer):
    def __init__(self, target_shape: tuple, input_shape: tuple | None = None):
        super().__init__()
        self.target_shape = target_shape
        self.input_shape = None

    def __str__(self) -> str:
        return f'Reshape(target_shape={self.target_shape}, input_shape={self.input_shape})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return np.reshape(input_data, (input_data.shape[0],) + self.target_shape)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return np.reshape(output_error, self.input_shape)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'target_shape': self.target_shape,
            'input_shape': self.input_shape
        }

    @staticmethod
    def from_config(config: dict):
        return Reshape(config['target_shape'], config['input_shape'])


class LSTMCell(Layer):
    def __init__(self, units: int, random_state: int | None = None):
        self.units = units
        self.random_state = random_state
        self.rng = np.random.default_rng(
            random_state if random_state is not None else int(time.time_ns()))

        self.EPSILON = 1e-7
        self.EPSILON_SIGMOID = 1e-4
        self.MAX_CLIP = 1e3

        self.Wf = None
        self.Uf = None
        self.bf = None
        self.Wi = None
        self.Ui = None
        self.bi = None
        self.Wc = None
        self.Uc = None
        self.bc = None
        self.Wo = None
        self.Uo = None
        self.bo = None

        self._init_gradients()
        self.cache = None

    def initialize_weights(self, input_dim: int):
        # Forget gate
        self.Wf = self.orthogonal_init((input_dim, self.units))
        self.Uf = self.orthogonal_init((self.units, self.units))
        self.bf = np.ones((1, self.units))

        # Input gate - slight negative bias for better selectivity
        self.Wi = self.orthogonal_init((input_dim, self.units))
        self.Ui = self.orthogonal_init((self.units, self.units))
        self.bi = -0.1 * np.ones((1, self.units))

        # Cell gate
        self.Wc = self.orthogonal_init((input_dim, self.units))
        self.Uc = self.orthogonal_init((self.units, self.units))
        self.bc = np.zeros((1, self.units))

        # Output gate
        self.Wo = self.orthogonal_init((input_dim, self.units))
        self.Uo = self.orthogonal_init((self.units, self.units))
        self.bo = np.zeros((1, self.units))

        self._init_gradients()

    def _init_gradients(self):
        if self.Wf is not None:
            self.dWf = np.zeros_like(self.Wf)
            self.dUf = np.zeros_like(self.Uf)
            self.dbf = np.zeros_like(self.bf)
            self.dWi = np.zeros_like(self.Wi)
            self.dUi = np.zeros_like(self.Ui)
            self.dbi = np.zeros_like(self.bi)
            self.dWc = np.zeros_like(self.Wc)
            self.dUc = np.zeros_like(self.Uc)
            self.dbc = np.zeros_like(self.bc)
            self.dWo = np.zeros_like(self.Wo)
            self.dUo = np.zeros_like(self.Uo)
            self.dbo = np.zeros_like(self.bo)

    def normalize_cell_state(self, c):
        norm = np.linalg.norm(c)
        if norm > self.MAX_CLIP:
            return c * (self.MAX_CLIP / (norm + self.EPSILON))
        return c

    def check_numerical_stability(self, x, name=""):
        if np.any(np.isnan(x)):
            raise ValueError(f"NaN detected in {name}")
        if np.any(np.isinf(x)):
            raise ValueError(f"Inf detected in {name}")
        return x

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.Wf is None:
            self.initialize_weights(x_t.shape[1])

        x_t = self.check_numerical_stability(x_t, "input")
        h_prev = self.check_numerical_stability(h_prev, "h_prev")
        c_prev = self.check_numerical_stability(c_prev, "c_prev")

        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

        self.f_gate_input = np.clip(
            np.dot(x_t, self.Wf) + np.dot(h_prev, self.Uf) + self.bf,
            -self.MAX_CLIP, self.MAX_CLIP
        )
        self.f_t = self.sigmoid(self.f_gate_input)

        self.i_gate_input = np.clip(
            np.dot(x_t, self.Wi) + np.dot(h_prev, self.Ui) + self.bi,
            -self.MAX_CLIP, self.MAX_CLIP
        )
        self.i_t = self.sigmoid(self.i_gate_input)

        self.c_gate_input = np.clip(
            np.dot(x_t, self.Wc) + np.dot(h_prev, self.Uc) + self.bc,
            -self.MAX_CLIP, self.MAX_CLIP
        )
        self.c_tilde = np.tanh(self.c_gate_input)

        self.c_t = self.f_t * c_prev + self.i_t * self.c_tilde
        self.c_t = self.normalize_cell_state(self.c_t)
        self.c_t = self.check_numerical_stability(self.c_t, "cell_state")

        self.o_gate_input = np.clip(
            np.dot(x_t, self.Wo) + np.dot(h_prev, self.Uo) + self.bo,
            -self.MAX_CLIP, self.MAX_CLIP
        )
        self.o_t = self.sigmoid(self.o_gate_input)

        self.c_t_tanh = np.tanh(self.c_t)
        self.h_t = self.o_t * self.c_t_tanh
        self.h_t = self.check_numerical_stability(self.h_t, "output")

        self.cache = {
            'x_t': self.x_t,
            'h_prev': self.h_prev,
            'c_prev': self.c_prev,
            'f_gate_input': self.f_gate_input,
            'i_gate_input': self.i_gate_input,
            'c_gate_input': self.c_gate_input,
            'o_gate_input': self.o_gate_input,
            'f_t': self.f_t,
            'i_t': self.i_t,
            'c_tilde': self.c_tilde,
            'c_t': self.c_t,
            'c_t_tanh': self.c_t_tanh,
            'o_t': self.o_t,
            'h_t': self.h_t
        }

        return self.h_t, self.c_t

    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dh_next = self.check_numerical_stability(dh_next, "dh_next")
        dc_next = self.check_numerical_stability(dc_next, "dc_next")

        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        c_prev = self.cache['c_prev']
        f_t = self.cache['f_t']
        i_t = self.cache['i_t']
        c_tilde = self.cache['c_tilde']
        o_t = self.cache['o_t']
        c_t = self.cache['c_t']
        c_t_tanh = self.cache['c_t_tanh']

        do = dh_next * c_t_tanh
        dc = dc_next + dh_next * o_t * (1 - c_t_tanh ** 2 + self.EPSILON)

        do_input = do * o_t * (1 - o_t + self.EPSILON)
        self.dWo += np.dot(x_t.T, do_input)
        self.dUo += np.dot(h_prev.T, do_input)
        self.dbo += np.sum(do_input, axis=0, keepdims=True)

        dc_prev = dc * f_t
        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i_t

        df_input = df * f_t * (1 - f_t + self.EPSILON)
        self.dWf += np.dot(x_t.T, df_input)
        self.dUf += np.dot(h_prev.T, df_input)
        self.dbf += np.sum(df_input, axis=0, keepdims=True)

        di_input = di * i_t * (1 - i_t + self.EPSILON)
        self.dWi += np.dot(x_t.T, di_input)
        self.dUi += np.dot(h_prev.T, di_input)
        self.dbi += np.sum(di_input, axis=0, keepdims=True)

        dc_tilde_input = dc_tilde * (1 - c_tilde ** 2 + self.EPSILON)
        self.dWc += np.dot(x_t.T, dc_tilde_input)
        self.dUc += np.dot(h_prev.T, dc_tilde_input)
        self.dbc += np.sum(dc_tilde_input, axis=0, keepdims=True)

        dx = (np.dot(df_input, self.Wf.T) +
              np.dot(di_input, self.Wi.T) +
              np.dot(dc_tilde_input, self.Wc.T) +
              np.dot(do_input, self.Wo.T))

        dh_prev = (np.dot(df_input, self.Uf.T) +
                   np.dot(di_input, self.Ui.T) +
                   np.dot(dc_tilde_input, self.Uc.T) +
                   np.dot(do_input, self.Uo.T))

        return self.check_numerical_stability(dx), self.check_numerical_stability(dh_prev), self.check_numerical_stability(dc_prev)

    def orthogonal_init(self, shape):
        if len(shape) < 2:
            return np.clip(self.rng.normal(0, 1, shape), -self.MAX_CLIP, self.MAX_CLIP)
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = self.rng.normal(0, 1, flat_shape)
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else vt
        q = q.reshape(shape)
        return np.clip(q, -self.MAX_CLIP, self.MAX_CLIP)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -self.MAX_CLIP, self.MAX_CLIP)
        result = 0.5 * (1 + np.tanh(x * 0.5))
        return np.clip(result, self.EPSILON_SIGMOID, 1 - self.EPSILON_SIGMOID)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'units': self.units,
            'random_state': self.random_state,
            'weights': {
                'Wf': self.Wf.tolist() if self.Wf is not None else None,
                'Uf': self.Uf.tolist() if self.Uf is not None else None,
                'bf': self.bf.tolist() if self.bf is not None else None,
                'Wi': self.Wi.tolist() if self.Wi is not None else None,
                'Ui': self.Ui.tolist() if self.Ui is not None else None,
                'bi': self.bi.tolist() if self.bi is not None else None,
                'Wc': self.Wc.tolist() if self.Wc is not None else None,
                'Uc': self.Uc.tolist() if self.Uc is not None else None,
                'bc': self.bc.tolist() if self.bc is not None else None,
                'Wo': self.Wo.tolist() if self.Wo is not None else None,
                'Uo': self.Uo.tolist() if self.Uo is not None else None,
                'bo': self.bo.tolist() if self.bo is not None else None
            }
        }

    @staticmethod
    def from_config(config: dict) -> 'LSTMCell':
        cell = LSTMCell(config['units'], config['random_state'])
        if config.get('weights'):
            w = config['weights']
            if w['Wf'] is not None:
                cell.Wf = np.array(w['Wf'])
                cell.Uf = np.array(w['Uf'])
                cell.bf = np.array(w['bf'])
                cell.Wi = np.array(w['Wi'])
                cell.Ui = np.array(w['Ui'])
                cell.bi = np.array(w['bi'])
                cell.Wc = np.array(w['Wc'])
                cell.Uc = np.array(w['Uc'])
                cell.bc = np.array(w['bc'])
                cell.Wo = np.array(w['Wo'])
                cell.Uo = np.array(w['Uo'])
                cell.bo = np.array(w['bo'])
        return cell


class LSTM(Layer):
    def __init__(self, units: int, return_sequences: bool = False, return_state: bool = False,
                 random_state: int | None = None, clip_value: float = 5.0, **kwargs):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.random_state = random_state
        self.clip_value = clip_value
        self.initialized = False
        self.cell = None
        self.last_h = None
        self.last_c = None
        self.cache = None
        self.input_shape = None
        self.EPSILON = 1e-7
        self.MAX_CLIP = 1e3

        for key, value in kwargs.items():
            setattr(self, key, value)

    def check_numerical_stability(self, x: np.ndarray, name: str = "") -> np.ndarray:
        if np.any(np.isnan(x)):
            raise ValueError(f"NaN detected in LSTM {name}")
        if np.any(np.isinf(x)):
            raise ValueError(f"Inf detected in LSTM {name}")
        return np.clip(x, -self.MAX_CLIP, self.MAX_CLIP)

    def normalize_gradients(self, gradients: dict) -> dict:
        total_norm = np.sqrt(sum(np.sum(g ** 2)
                             for g in gradients.values()) + self.EPSILON)
        scaling_factor = min(1.0, self.clip_value /
                             (total_norm + self.EPSILON))
        return {k: v * scaling_factor for k, v in gradients.items()}

    def forward_pass(self, x: np.ndarray, training: bool = True) -> np.ndarray | tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, timesteps, features), got shape {x.shape}")

        self.input_shape = x.shape
        batch_size, timesteps, input_dim = x.shape

        x = self.check_numerical_stability(x, "input")

        if not hasattr(self, '_zeros_template') or self._zeros_template.shape[0] != batch_size:
            self._zeros_template = np.zeros((batch_size, self.units))

        h = self._zeros_template.copy()
        c = self._zeros_template.copy()

        if not self.initialized:
            self.cell = LSTMCell(self.units, self.random_state)
            self.cell.initialize_weights(input_dim)
            self.initialized = True

        all_h = []
        all_c = []
        self.cache = []

        for t in range(timesteps):
            x_t = x[:, t, :]
            h, c = self.cell.forward(x_t, h, c)
            h = self.check_numerical_stability(h, f"hidden_state_t{t}")
            c = self.check_numerical_stability(c, f"cell_state_t{t}")

            if self.return_sequences:
                all_h.append(h)
            all_c.append(c)
            if training:
                self.cache.append(self.cell.cache)

        self.last_h = h
        self.last_c = c

        if self.return_sequences:
            all_h = np.stack(all_h, axis=1)
            if self.return_state:
                return all_h, self.last_h, self.last_c
            return all_h
        else:
            if self.return_state:
                return self.last_h, self.last_h, self.last_c
            return self.last_h

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        output_error = self.check_numerical_stability(
            output_error, "output_error")
        batch_size, timesteps, input_dim = self.input_shape

        if len(output_error.shape) == 2:
            full_dout = np.zeros((batch_size, timesteps, self.units))
            full_dout[:, -1, :] = output_error
            output_error = full_dout

        dx = np.zeros((batch_size, timesteps, input_dim))
        dh_next = np.zeros((batch_size, self.units))
        dc_next = np.zeros((batch_size, self.units))

        self.cell._init_gradients()
        all_gradients = {}

        for t in reversed(range(timesteps)):
            dh = output_error[:, t, :] + dh_next
            dh = self.check_numerical_stability(dh, f"dh_t{t}")

            self.cell.cache = self.cache[t]
            dx_t, dh_next, dc_next = self.cell.backward(dh, dc_next)

            dx[:, t, :] = dx_t

            # Collect gradients for normalization
            t_gradients = {
                f'dWf_{t}': self.cell.dWf,
                f'dUf_{t}': self.cell.dUf,
                f'dbf_{t}': self.cell.dbf,
                f'dWi_{t}': self.cell.dWi,
                f'dUi_{t}': self.cell.dUi,
                f'dbi_{t}': self.cell.dbi,
                f'dWc_{t}': self.cell.dWc,
                f'dUc_{t}': self.cell.dUc,
                f'dbc_{t}': self.cell.dbc,
                f'dWo_{t}': self.cell.dWo,
                f'dUo_{t}': self.cell.dUo,
                f'dbo_{t}': self.cell.dbo,
                f'dx_{t}': dx_t
            }

            all_gradients.update(t_gradients)

        # Normalize all gradients together
        normalized_gradients = self.normalize_gradients(all_gradients)

        # Update cell gradients and dx with normalized values
        for t in range(timesteps):
            dx[:, t, :] = normalized_gradients[f'dx_{t}']
            if t == 0:  # Update cell gradients only once
                self.cell.dWf = normalized_gradients[f'dWf_{t}']
                self.cell.dUf = normalized_gradients[f'dUf_{t}']
                self.cell.dbf = normalized_gradients[f'dbf_{t}']
                self.cell.dWi = normalized_gradients[f'dWi_{t}']
                self.cell.dUi = normalized_gradients[f'dUi_{t}']
                self.cell.dbi = normalized_gradients[f'dbi_{t}']
                self.cell.dWc = normalized_gradients[f'dWc_{t}']
                self.cell.dUc = normalized_gradients[f'dUc_{t}']
                self.cell.dbc = normalized_gradients[f'dbc_{t}']
                self.cell.dWo = normalized_gradients[f'dWo_{t}']
                self.cell.dUo = normalized_gradients[f'dUo_{t}']
                self.cell.dbo = normalized_gradients[f'dbo_{t}']

        return self.check_numerical_stability(dx, "final_dx")

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'units': self.units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'clip_value': self.clip_value,
            'random_state': self.random_state,
            'cell_config': self.cell.get_config() if self.cell is not None else None
        }

    @staticmethod
    def from_config(config: dict) -> 'LSTM':
        lstm = LSTM(
            config['units'],
            config['return_sequences'],
            config.get('return_state', False),
            config['random_state'],
            config.get('clip_value', 5.0)
        )
        if config.get('cell_config'):
            lstm.cell = LSTMCell.from_config(config['cell_config'])
            lstm.initialized = True
        return lstm

    def __str__(self):
        return (f'LSTM(units={self.units}, '
                f'return_sequences={self.return_sequences}, '
                f'return_state={self.return_state}, '
                f'clip_value={self.clip_value}, '
                f'random_state={self.random_state})')


class Bidirectional(Layer):
    def __init__(self, layer: LSTM):
        super().__init__()
        if not isinstance(layer, LSTM):
            raise ValueError("Bidirectional layer only supports LSTM layers")

        self.forward_layer = layer
        self.backward_layer = LSTM(
            layer.units,
            layer.return_sequences,
            layer.return_state,
            layer.random_state
        )

    def __str__(self) -> str:
        return f'Bidirectional(layer={str(self.forward_layer)})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.forward_output = self.forward_layer.forward_pass(
            input_data, training)

        backward_input = input_data[:, ::-1, :]
        self.backward_output = self.backward_layer.forward_pass(
            backward_input, training)

        if isinstance(self.forward_output, tuple):
            if self.forward_layer.return_sequences:
                forward_seq, forward_h, forward_c = self.forward_output
                backward_seq, backward_h, backward_c = self.backward_output

                backward_seq = backward_seq[:, ::-1, :]

                combined_seq = np.concatenate(
                    [forward_seq, backward_seq], axis=-1)
                combined_h = np.concatenate([forward_h, backward_h], axis=-1)
                combined_c = np.concatenate([forward_c, backward_c], axis=-1)

                return combined_seq, combined_h, combined_c
            else:
                forward_h, _, forward_c = self.forward_output
                backward_h, _, backward_c = self.backward_output
                combined_h = np.concatenate([forward_h, backward_h], axis=-1)
                combined_c = np.concatenate([forward_c, backward_c], axis=-1)
                return combined_h, combined_h, combined_c
        else:
            if self.forward_layer.return_sequences:
                backward_seq = self.backward_output[:, ::-1, :]
                return np.concatenate([self.forward_output, backward_seq], axis=-1)
            else:
                return np.concatenate([self.forward_output, self.backward_output], axis=-1)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        forward_dim = output_error.shape[-1] // 2

        if len(output_error.shape) == 3:
            forward_error = output_error[:, :, :forward_dim]
            backward_error = output_error[:, :, forward_dim:]

            backward_error = backward_error[:, ::-1, :]
        else:
            forward_error = output_error[:, :forward_dim]
            backward_error = output_error[:, forward_dim:]

        forward_dx = self.forward_layer.backward_pass(forward_error)
        backward_dx = self.backward_layer.backward_pass(backward_error)

        if len(output_error.shape) == 3:
            backward_dx = backward_dx[:, ::-1, :]

        return forward_dx + backward_dx

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'forward_layer': self.forward_layer.get_config(),
            'backward_layer': self.backward_layer.get_config()
        }

    @staticmethod
    def from_config(config: dict) -> 'Bidirectional':
        forward_layer = Layer.from_config(config['forward_layer'])
        layer = Bidirectional(forward_layer)
        layer.backward_layer = Layer.from_config(config['backward_layer'])
        return layer


class Unidirectional(Layer):
    """Wrapper class that makes it explicit that a layer processes sequences in one direction"""

    def __init__(self, layer: LSTM):
        super().__init__()
        if not isinstance(layer, LSTM):
            raise ValueError("Unidirectional layer only supports LSTM layers")
        self.layer = layer

    def __str__(self) -> str:
        return f'Unidirectional(layer={str(self.layer)})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        return self.layer.forward_pass(input_data, training)

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        return self.layer.backward_pass(output_error)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'layer': self.layer.get_config()
        }

    @staticmethod
    def from_config(config: dict) -> 'Unidirectional':
        layer = Layer.from_config(config['layer'])
        return Unidirectional(layer)


class GRUCell:
    def __init__(self, units: int, random_state: int | None = None, clip_value: float = 5.0):
        self.units = units
        self.random_state = random_state
        self.clip_value = clip_value
        self.rng = np.random.default_rng(
            random_state if random_state is not None else int(time.time_ns()))

        # Reset gate weights
        self.Wr = None
        self.Ur = None
        self.br = None

        # Update gate weights
        self.Wz = None
        self.Uz = None
        self.bz = None

        # Candidate state weights
        self.Wh = None
        self.Uh = None
        self.bh = None

        self._init_gradients()
        self.cache = None

    def initialize_weights(self, input_dim: int):
        # Reset gate
        self.Wr = self.orthogonal_init((input_dim, self.units))
        self.Ur = self.orthogonal_init((self.units, self.units))
        self.br = np.zeros((1, self.units))

        # Update gate
        self.Wz = self.orthogonal_init((input_dim, self.units))
        self.Uz = self.orthogonal_init((self.units, self.units))
        self.bz = np.zeros((1, self.units))

        # Candidate state
        self.Wh = self.orthogonal_init((input_dim, self.units))
        self.Uh = self.orthogonal_init((self.units, self.units))
        self.bh = np.zeros((1, self.units))

        # Initialize gradients
        self._init_gradients()

    def _init_gradients(self):
        if self.Wr is not None:
            self.dWr = np.zeros_like(self.Wr)
            self.dUr = np.zeros_like(self.Ur)
            self.dbr = np.zeros_like(self.br)

            self.dWz = np.zeros_like(self.Wz)
            self.dUz = np.zeros_like(self.Uz)
            self.dbz = np.zeros_like(self.bz)

            self.dWh = np.zeros_like(self.Wh)
            self.dUh = np.zeros_like(self.Uh)
            self.dbh = np.zeros_like(self.bh)

    def orthogonal_init(self, shape):
        if len(shape) < 2:
            return self.rng.normal(0, 1, shape)
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = self.rng.normal(0, 1, flat_shape)
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else vt
        q = q.reshape(shape)
        return q

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        if self.Wr is None:
            self.initialize_weights(x_t.shape[1])

        # Store inputs for backprop
        self.x_t = x_t
        self.h_prev = h_prev

        # Reset gate
        self.r_gate_input = np.dot(x_t, self.Wr) + \
            np.dot(h_prev, self.Ur) + self.br
        self.r_t = self.sigmoid(self.r_gate_input)

        # Update gate
        self.z_gate_input = np.dot(x_t, self.Wz) + \
            np.dot(h_prev, self.Uz) + self.bz
        self.z_t = self.sigmoid(self.z_gate_input)

        # Candidate state
        self.h_candidate_input = np.dot(
            x_t, self.Wh) + np.dot(self.r_t * h_prev, self.Uh) + self.bh
        self.h_candidate = np.tanh(self.h_candidate_input)

        # New hidden state
        self.h_t = (1 - self.z_t) * h_prev + self.z_t * self.h_candidate

        self.cache = {
            'x_t': self.x_t,
            'h_prev': self.h_prev,
            'r_gate_input': self.r_gate_input,
            'z_gate_input': self.z_gate_input,
            'h_candidate_input': self.h_candidate_input,
            'r_t': self.r_t,
            'z_t': self.z_t,
            'h_candidate': self.h_candidate,
            'h_t': self.h_t
        }

        return self.h_t

    def backward(self, dh_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        r_t = self.cache['r_t']
        z_t = self.cache['z_t']
        h_candidate = self.cache['h_candidate']

        # Clip incoming gradients
        dh_next = self.clip_gradients(dh_next)

        # Gradients for gates
        dh_candidate = dh_next * z_t
        dz = dh_next * (h_candidate - h_prev)
        dh_prev = dh_next * (1 - z_t)

        # Candidate state gradients
        dh_candidate_input = dh_candidate * (1 - h_candidate ** 2)
        dh_candidate_input = self.clip_gradients(dh_candidate_input)

        self.dWh += np.dot(x_t.T, dh_candidate_input)
        self.dUh += np.dot((r_t * h_prev).T, dh_candidate_input)
        self.dbh += np.sum(dh_candidate_input, axis=0, keepdims=True)

        dr_h_prev = np.dot(dh_candidate_input, self.Uh.T)
        dr = dr_h_prev * h_prev
        dh_prev += dr_h_prev * r_t

        # Update gate gradients
        dz_input = dz * z_t * (1 - z_t)
        dz_input = self.clip_gradients(dz_input)

        self.dWz += np.dot(x_t.T, dz_input)
        self.dUz += np.dot(h_prev.T, dz_input)
        self.dbz += np.sum(dz_input, axis=0, keepdims=True)

        # Reset gate gradients
        dr_input = dr * r_t * (1 - r_t)
        dr_input = self.clip_gradients(dr_input)

        self.dWr += np.dot(x_t.T, dr_input)
        self.dUr += np.dot(h_prev.T, dr_input)
        self.dbr += np.sum(dr_input, axis=0, keepdims=True)

        dx = (np.dot(dz_input, self.Wz.T) +
              np.dot(dr_input, self.Wr.T) +
              np.dot(dh_candidate_input, self.Wh.T))

        dh_prev += (np.dot(dz_input, self.Uz.T) +
                    np.dot(dr_input, self.Ur.T))

        dx = self.clip_gradients(dx)
        dh_prev = self.clip_gradients(dh_prev)

        return dx, dh_prev

    def clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        return np.clip(gradient, -self.clip_value, self.clip_value)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        result = 0.5 * (1 + np.tanh(x * 0.5))
        return np.clip(result, EPSILON_SIGMOID, 1 - EPSILON_SIGMOID)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'units': self.units,
            'random_state': self.random_state,
            'clip_value': self.clip_value,
            'weights': {
                'Wr': self.Wr.tolist() if self.Wr is not None else None,
                'Ur': self.Ur.tolist() if self.Ur is not None else None,
                'br': self.br.tolist() if self.br is not None else None,
                'Wz': self.Wz.tolist() if self.Wz is not None else None,
                'Uz': self.Uz.tolist() if self.Uz is not None else None,
                'bz': self.bz.tolist() if self.bz is not None else None,
                'Wh': self.Wh.tolist() if self.Wh is not None else None,
                'Uh': self.Uh.tolist() if self.Uh is not None else None,
                'bh': self.bh.tolist() if self.bh is not None else None
            }
        }

    @staticmethod
    def from_config(config: dict) -> 'GRUCell':
        cell = GRUCell(config['units'], config['random_state'],
                       config.get('clip_value', 5.0))
        if config.get('weights'):
            w = config['weights']
            if w['Wr'] is not None:
                cell.Wr = np.array(w['Wr'])
                cell.Ur = np.array(w['Ur'])
                cell.br = np.array(w['br'])
                cell.Wz = np.array(w['Wz'])
                cell.Uz = np.array(w['Uz'])
                cell.bz = np.array(w['bz'])
                cell.Wh = np.array(w['Wh'])
                cell.Uh = np.array(w['Uh'])
                cell.bh = np.array(w['bh'])
        return cell


class GRU(Layer):
    def __init__(self, units: int, return_sequences: bool = False, random_state: int | None = None,
                 clip_value: float = 5.0, **kwargs):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.random_state = random_state
        self.initialized = False
        self.cell = None
        self.last_h = None
        self.cache = None
        self.input_shape = None
        self.clip_value = clip_value

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f'GRU(units={self.units}, return_sequences={self.return_sequences}, random_state={self.random_state}, clip_value={self.clip_value})'

    def forward_pass(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, timesteps, features), got shape {x.shape}")
        self.input_shape = x.shape
        batch_size, timesteps, input_dim = x.shape

        if not hasattr(self, '_zeros_template') or self._zeros_template.shape[0] != batch_size:
            self._zeros_template = np.zeros((batch_size, self.units))

        h = self._zeros_template.copy()

        if not self.initialized:
            self.cell = GRUCell(self.units, self.random_state, self.clip_value)
            self.cell.initialize_weights(input_dim)
            self.initialized = True

        all_h = []
        self.cache = []

        for t in range(timesteps):
            x_t = x[:, t, :]
            h = self.cell.forward(x_t, h)
            if self.return_sequences:
                all_h.append(h)
            if training:
                self.cache.append(self.cell.cache)

        self.last_h = h

        if self.return_sequences:
            return np.stack(all_h, axis=1)
        return self.last_h

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        batch_size, timesteps, input_dim = self.input_shape

        if len(output_error.shape) == 2:
            full_dout = np.zeros((batch_size, timesteps, self.units))
            full_dout[:, -1, :] = output_error
            output_error = full_dout

        dx = np.zeros((batch_size, timesteps, input_dim))
        dh_next = np.zeros((batch_size, self.units))

        self.cell._init_gradients()

        squared_norm_sum = 0.0

        for t in reversed(range(timesteps)):
            dh = output_error[:, t, :] + dh_next

            self.cell.cache = self.cache[t]
            dx_t, dh_next = self.cell.backward(dh)
            dx[:, t, :] = dx_t

            squared_norm_sum += (np.sum(dx_t ** 2) +
                                 np.sum(self.cell.dWr ** 2) + np.sum(self.cell.dUr ** 2) + np.sum(self.cell.dbr ** 2) +
                                 np.sum(self.cell.dWz ** 2) + np.sum(self.cell.dUz ** 2) + np.sum(self.cell.dbz ** 2) +
                                 np.sum(self.cell.dWh ** 2) + np.sum(self.cell.dUh ** 2) + np.sum(self.cell.dbh ** 2))

        global_norm = np.sqrt(squared_norm_sum)
        scaling_factor = min(1.0, self.clip_value / (global_norm + 1e-8))
        if scaling_factor < 1.0:
            dx *= scaling_factor
            for grad in self.cell.__dict__:
                if grad.startswith('d'):
                    setattr(self.cell, grad, getattr(
                        self.cell, grad) * scaling_factor)

        return dx

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'units': self.units,
            'return_sequences': self.return_sequences,
            'random_state': self.random_state,
            'clip_value': self.clip_value,
            'cell_config': self.cell.get_config() if self.cell is not None else None
        }

    @staticmethod
    def from_config(config: dict) -> 'GRU':
        gru = GRU(
            config['units'],
            config['return_sequences'],
            config['random_state'],
            config.get('clip_value', 5.0)
        )
        if config.get('cell_config'):
            gru.cell = GRUCell.from_config(config['cell_config'])
            gru.initialized = True
        return gru


class Attention(Layer):
    def __init__(self, use_scale: bool = True, score_mode: str = "dot", return_sequences: bool = False):
        super().__init__()
        # for now cuz I'm too lazy to implement the others
        valid_score_modes = ["dot"]
        if score_mode not in valid_score_modes:
            raise ValueError(
                f"score_mode must be one of {valid_score_modes}, got {score_mode}")
        self.use_scale = use_scale
        self.score_mode = score_mode
        self.return_sequences = return_sequences
        self.cache = {}

    def __str__(self) -> str:
        return f'Attention(use_scale={self.use_scale}, score_mode={self.score_mode}, return_sequences={self.return_sequences})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        batch_size, seq_length, features = input_data.shape

        self.cache['input_shape'] = input_data.shape
        self.cache['input'] = input_data
        scores = np.zeros((batch_size, seq_length, seq_length))

        for i in range(batch_size):
            if self.score_mode == "dot":
                scores[i] = np.dot(input_data[i], input_data[i].T)
                if self.use_scale:
                    scores[i] *= 1.0 / np.sqrt(features)

        attention_weights = np.zeros_like(scores)
        for i in range(batch_size):
            attention_weights[i] = self._softmax(scores[i])

        self.cache['attention_weights'] = attention_weights

        context = np.zeros_like(input_data)
        for i in range(batch_size):
            context[i] = np.dot(attention_weights[i], input_data[i])

        if not self.return_sequences:
            context = np.mean(context, axis=1)
        return context

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        batch_size, seq_length, features = self.cache['input_shape']
        attention_weights = self.cache['attention_weights']
        input_data = self.cache['input']

        if not self.return_sequences:
            output_error = np.expand_dims(output_error, 1)
            output_error = np.repeat(output_error, seq_length, axis=1)

        d_input = np.zeros((batch_size, seq_length, features))

        for i in range(batch_size):
            d_context = output_error[i]

            d_weights = np.dot(d_context, input_data[i].T)

            d_scores = d_weights * attention_weights[i]
            d_scores -= attention_weights[i] * np.sum(
                d_weights * attention_weights[i], axis=-1, keepdims=True)

            if self.use_scale:
                d_scores *= 1.0 / np.sqrt(features)

            d_input[i] = np.dot(attention_weights[i].T, d_context)

            if self.score_mode == "dot":
                d_scores_sym = (d_scores + d_scores.T) / 2
                d_input[i] += np.dot(d_scores_sym, input_data[i])

        self.cache.clear()
        return d_input

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'use_scale': self.use_scale,
            'score_mode': self.score_mode,
            'return_sequences': self.return_sequences
        }

    @staticmethod
    def from_config(config: dict):
        return Attention(
            use_scale=config['use_scale'],
            score_mode=config['score_mode'],
            return_sequences=config.get('return_sequences', False)
        )


class Conv2DTranspose(Layer):
    def __init__(self, filters: int, kernel_size: int | tuple, strides: int | tuple = 1,
                 padding: str = 'valid', weights_init: str = "default", bias_init: str = "default",
                 random_state: int = None, **kwargs):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
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
        _, _, _, in_channels = input_shape
        
        self.rng = np.random.default_rng(
            self.random_state if self.random_state is not None else int(time.time_ns()))
        
        if self.weights_init == "xavier":
            self.weights = self.rng.normal(0, np.sqrt(2 / (np.prod(self.kernel_size) * self.filters)),
                                        (*self.kernel_size, self.filters, in_channels))
        elif self.weights_init == "he":
            self.weights = self.rng.normal(0, np.sqrt(2 / (in_channels * np.prod(self.kernel_size))),
                                        (*self.kernel_size, self.filters, in_channels))
        elif self.weights_init == "default":
            self.weights = self.rng.normal(0, 0.01, (*self.kernel_size, self.filters, in_channels))
        else:
            raise ValueError("Invalid weights_init value. Possible values are 'xavier', 'he', and 'default'.")
            
        if self.bias_init == "default":
            self.bias = np.zeros((1, 1, 1, self.filters))
        elif self.bias_init == "normal":
            self.bias = self.rng.normal(0, 0.01, (1, 1, 1, self.filters))
        elif self.bias_init == "uniform":
            self.bias = self.rng.uniform(-0.1, 0.1, (1, 1, 1, self.filters))
        elif self.bias_init == "small":
            self.bias = np.full((1, 1, 1, self.filters), 0.01)
        else:
            raise ValueError("Invalid bias_init value.")
            
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            assert len(input_data.shape) == 4, "Conv2DTranspose input must be 4D (batch_size, height, width, channels)"
            self.initialize_weights(input_data.shape)
            
        self.input = input_data
        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_height, kernel_width, out_channels, _ = self.weights.shape
        
        if self.padding == 'same':
            out_height = in_height * self.strides[0]
            out_width = in_width * self.strides[1]
            pad_height = max((in_height - 1) * self.strides[0] + kernel_height - out_height, 0)
            pad_width = max((in_width - 1) * self.strides[1] + kernel_width - out_width, 0)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        else:
            out_height = (in_height - 1) * self.strides[0] + kernel_height
            out_width = (in_width - 1) * self.strides[1] + kernel_width
            pad_top = pad_bottom = pad_left = pad_right = 0

        padded_output = np.zeros((batch_size, out_height + pad_top + pad_bottom,
                                out_width + pad_left + pad_right, out_channels))

        for h in range(in_height):
            for w in range(in_width):
                h_start = h * self.strides[0]
                w_start = w * self.strides[1]
                
                out_slice = padded_output[:, h_start:h_start + kernel_height,
                                        w_start:w_start + kernel_width, :]
                
                for c in range(in_channels):
                    weight_slice = self.weights[:, :, :, c]
                    input_val = input_data[:, h, w, c:c+1]
                    out_slice += np.expand_dims(weight_slice, 0) * np.expand_dims(input_val, (1, 2))

        if self.padding == 'valid':
            output = padded_output
        else:
            output = padded_output[:, pad_top:pad_top + out_height,
                                 pad_left:pad_left + out_width, :]

        return output + self.bias

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        batch_size = output_error.shape[0]
        kernel_height, kernel_width, out_channels, in_channels = self.weights.shape

        d_input = np.zeros_like(self.input)
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.sum(output_error, axis=(0, 1, 2), keepdims=True)

        for h in range(d_input.shape[1]):
            for w in range(d_input.shape[2]):
                h_start = h * self.strides[0]
                w_start = w * self.strides[1]
                
                error_field = output_error[:, h_start:h_start + kernel_height,
                                        w_start:w_start + kernel_width, :]
                
                if error_field.shape[1:3] == (kernel_height, kernel_width):
                    for c in range(in_channels):
                        weight_slice = self.weights[:, :, :, c]
                        d_input[:, h, w, c] = np.sum(error_field * weight_slice, axis=(1, 2, 3))
                        
                        for b in range(batch_size):
                            self.d_weights[:, :, :, c] += error_field[b] * self.input[b, h, w, c]

        return d_input

    def __str__(self) -> str:
        return f'Conv2DTranspose(filters={self.filters}, kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding})'

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'weights_init': self.weights_init,
            'bias_init': self.bias_init,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict):
        layer = Conv2DTranspose(
            config['filters'],
            config['kernel_size'],
            config['strides'],
            config['padding'],
            config['weights_init'],
            config['bias_init'],
            config['random_state']
        )
        if config['weights'] is not None:
            layer.weights = np.array(config['weights'])
            layer.bias = np.array(config['bias'])
        return layer


class UpSampling2D(Layer):
    def __init__(self, size=(2, 2), interpolation="nearest", **kwargs):
        super().__init__()
        self.size = tuple(size) if isinstance(
            size, (list, tuple)) else (size, size)
        self.interpolation = interpolation

        if not isinstance(self.size, tuple):
            raise TypeError('Size must be a tuple or list of 2 integers.')
        if len(self.size) != 2:
            raise ValueError('Size must have exactly 2 elements.')
        if not all(isinstance(s, (int, np.integer)) and s > 0 for s in self.size):
            raise ValueError('Size elements must be positive integers.')

        if interpolation not in ['nearest', 'bilinear', 'bicubic']:
            raise ValueError(
                'interpolation must be one of "nearest", "bilinear", or "bicubic"')

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f'UpSampling2D(size={self.size}, interpolation={self.interpolation})'

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data

        batch_size, height, width, channels = input_data.shape
        height_factor, width_factor = self.size

        if self.interpolation == 'nearest':
            output = np.repeat(
                np.repeat(input_data, height_factor, axis=1), width_factor, axis=2)

        elif self.interpolation in ['bilinear', 'bicubic']:
            output_height = height * height_factor
            output_width = width * width_factor

            y = np.linspace(0, height - 1, output_height)
            x = np.linspace(0, width - 1, output_width)
            x_grid, y_grid = np.meshgrid(x, y)

            y0 = np.floor(y_grid).astype(int)
            x0 = np.floor(x_grid).astype(int)
            y1 = np.minimum(y0 + 1, height - 1)
            x1 = np.minimum(x0 + 1, width - 1)

            wy = y_grid - y0
            wx = x_grid - x0

            wy = wy[:, :, np.newaxis]
            wx = wx[:, :, np.newaxis]

            output = np.zeros((batch_size, output_height,
                              output_width, channels), dtype=input_data.dtype)

            for b in range(batch_size):
                top = (1 - wx) * input_data[b, y0,
                                            x0] + wx * input_data[b, y0, x1]
                bottom = (1 - wx) * \
                    input_data[b, y1, x0] + wx * input_data[b, y1, x1]
                output[b] = (1 - wy) * top + wy * bottom

        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        batch_size, height, width, channels = self.input.shape
        height_factor, width_factor = self.size

        if self.interpolation == 'nearest':
            output_error_reshaped = output_error.reshape(
                batch_size,
                height, height_factor,
                width, width_factor,
                channels
            )
            input_error = output_error_reshaped.sum(axis=(2, 4))

        else:  # bilinear
            output_height = height * height_factor
            output_width = width * width_factor

            y = np.linspace(0, height - 1, output_height)
            x = np.linspace(0, width - 1, output_width)
            x_grid, y_grid = np.meshgrid(x, y)

            y0 = np.floor(y_grid).astype(int)
            x0 = np.floor(x_grid).astype(int)
            y1 = np.minimum(y0 + 1, height - 1)
            x1 = np.minimum(x0 + 1, width - 1)

            wy = (y_grid - y0)[:, :, np.newaxis]
            wx = (x_grid - x0)[:, :, np.newaxis]

            input_error = np.zeros_like(self.input)

            for b in range(batch_size):
                input_error[b, y0, x0] += ((1 - wy) * (1 - wx)
                                           * output_error[b]).sum(axis=(0, 1))
                input_error[b, y0, x1] += ((1 - wy) *
                                           wx * output_error[b]).sum(axis=(0, 1))
                input_error[b, y1, x0] += (wy * (1 - wx)
                                           * output_error[b]).sum(axis=(0, 1))
                input_error[b, y1, x1] += (wy * wx *
                                           output_error[b]).sum(axis=(0, 1))

        return input_error

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'size': self.size,
            'interpolation': self.interpolation
        }

    @staticmethod
    def from_config(config: dict) -> 'UpSampling2D':
        return UpSampling2D(
            size=config['size'],
            interpolation=config['interpolation']
        )


class MultiHeadAttention(Layer):
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        output_shape: int = None,
        attention_axes: list[int] = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        normalize_attention: bool = False,
        random_state: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.key_dim: int = key_dim
        self.value_dim: int = value_dim if value_dim else key_dim
        self.dropout_rate: float = dropout_rate
        self.use_bias: bool = use_bias
        self.output_shape: int = output_shape
        self.attention_axes: list[int] = attention_axes
        self.kernel_initializer: str = kernel_initializer
        self.bias_initializer: str = bias_initializer
        self.random_state: int = random_state
        self.normalize_attention: bool = normalize_attention

        self.query_dense: Dense = None
        self.key_dense: Dense = None
        self.value_dense: Dense = None
        self.output_dense: Dense = None
        self.dropout: Dropout = Dropout(
            dropout_rate, random_state=random_state) if dropout_rate > 0 else None

        self.scale = 1.0 / np.sqrt(self.key_dim)

        self.attention_weights: np.ndarray | None = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f'MultiHeadAttention(num_heads={self.num_heads}, key_dim={self.key_dim})'

    def build_dense_layer(self, units: int, input_shape: tuple[int, ...]) -> Dense:
        return Dense(
            units=units,
            weights_init=self.kernel_initializer,
            bias_init=self.bias_initializer if self.use_bias else None,
            random_state=self.random_state
        )

    def initialize_weights(self, input_shape: tuple[int, ...]) -> None:
        embedding_dim: int = input_shape[-1]

        if self.query_dense is None:
            self.query_dense = self.build_dense_layer(
                self.num_heads * self.key_dim, input_shape)
            self.key_dense = self.build_dense_layer(
                self.num_heads * self.key_dim, input_shape)
            self.value_dense = self.build_dense_layer(
                self.num_heads * self.value_dim, input_shape)

        output_dim: int = self.output_shape if self.output_shape else embedding_dim
        if self.output_dense is None:
            self.output_dense = self.build_dense_layer(output_dim, input_shape)

    def _reshape_for_attention(self, x: np.ndarray, batch_size: int, seq_length: int) -> np.ndarray:
        x = np.reshape(x, (batch_size, seq_length, self.num_heads, -1))
        return np.transpose(x, (0, 2, 1, 3))

    def _scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray,
                                      value: np.ndarray, mask: np.ndarray = None,
                                      training: bool = True) -> np.ndarray:

        if self.normalize_attention:
            query_norm = np.sqrt(
                np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
            key_norm = np.sqrt(
                np.sum(key * key, axis=-1, keepdims=True) + 1e-6)

            query_normalized = query / query_norm
            key_normalized = key / key_norm

            matmul_qk = np.matmul(query_normalized, np.transpose(
                key_normalized, (0, 1, 3, 2)))
        else:
            matmul_qk = np.matmul(query, np.transpose(key, (0, 1, 3, 2)))

        d_k = key.shape[-1]

        scaling_factor = np.sqrt(d_k)
        scaled_attention_logits = matmul_qk / scaling_factor

        MASKING_VALUE = -1e9
        if mask is not None:
            scaled_attention_logits += (mask * MASKING_VALUE)

        attention_weights = self._softmax_with_mask(
            scaled_attention_logits, mask)
        self.attention_weights = attention_weights

        output = np.matmul(attention_weights, value)

        return output

    def _softmax_with_mask(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is not None:
            x_masked = np.where(mask, -1e9, x)
            max_x = np.max(x_masked, axis=-1, keepdims=True)
            exp_x = np.exp(x_masked - max_x)
            exp_x = np.where(mask, 0.0, exp_x)
        else:
            max_x = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - max_x)

        sum_x = np.sum(exp_x, axis=-1, keepdims=True)
        sum_x = np.maximum(sum_x, 1e-6)

        return exp_x / sum_x

    def forward_pass(self, inputs: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray], mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        if isinstance(inputs, tuple):
            query, key, value = inputs
            self.is_cross_attention = True
        else:
            query = key = value = inputs
            self.is_cross_attention = False

        batch_size = query.shape[0]

        self.query_input = query
        self.key_input = key
        self.value_input = value

        if self.query_dense is None:
            self.initialize_weights(query.shape)

        Q = self.query_dense.forward_pass(query)
        K = self.key_dense.forward_pass(key)
        V = self.value_dense.forward_pass(value)

        self.reshaped_query = self._reshape_for_attention(
            Q, batch_size, query.shape[1])

        self.reshaped_key = self._reshape_for_attention(
            K, batch_size, key.shape[1])
        self.reshaped_value = self._reshape_for_attention(
            V, batch_size, value.shape[1])

        scaled_attention = self._scaled_dot_product_attention(
            self.reshaped_query,
            self.reshaped_key,
            self.reshaped_value,
            mask,
            training
        )

        scaled_attention = np.transpose(scaled_attention, (0, 2, 1, 3))
        concat_attention = np.reshape(scaled_attention,
                                      (batch_size, -1, self.num_heads * self.value_dim))

        output = self.output_dense.forward_pass(concat_attention)
        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = output_error.shape[0]
        query_seq_length = output_error.shape[1]
        key_value_seq_length = self.reshaped_value.shape[2]

        d_attention_output = np.reshape(output_error,
                                        (batch_size, query_seq_length, self.num_heads, -1))
        d_attention_output = normalize_gradient(d_attention_output)
        d_attention_output = np.transpose(d_attention_output, (0, 2, 1, 3))

        d_attention = np.matmul(d_attention_output, np.transpose(
            self.reshaped_value, (0, 1, 3, 2)))
        attention_probs = self.attention_weights
        dot = np.sum(d_attention * attention_probs, axis=-1, keepdims=True)
        d_attention_probs = d_attention - dot
        d_attention_probs = d_attention_probs * attention_probs

        d_values = np.matmul(attention_probs.transpose(
            0, 1, 3, 2), d_attention_output)
        d_query = np.matmul(d_attention_probs, self.reshaped_key)
        d_key = np.matmul(d_attention_probs.transpose(
            0, 1, 3, 2), self.reshaped_query)

        d_values = normalize_gradient(d_values)
        d_query = normalize_gradient(d_query)
        d_key = normalize_gradient(d_key)

        d_query = np.transpose(d_query, (0, 2, 1, 3))
        d_key = np.transpose(d_key, (0, 2, 1, 3))
        d_values = np.transpose(d_values, (0, 2, 1, 3))

        final_dim = self.num_heads * self.key_dim
        d_query = np.reshape(
            d_query, (batch_size, query_seq_length, final_dim))
        d_key = np.reshape(
            d_key, (batch_size, key_value_seq_length, final_dim))
        d_values = np.reshape(
            d_values, (batch_size, key_value_seq_length, final_dim))

        d_query = self.query_dense.backward_pass(d_query)
        d_key = self.key_dense.backward_pass(d_key)
        d_value = self.value_dense.backward_pass(d_values)

        if self.is_cross_attention:
            return d_query, d_key, d_value
        return d_query

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x: np.ndarray = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'output_shape': self.output_shape,
            'attention_axes': self.attention_axes,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict) -> "MultiHeadAttention":
        return MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['key_dim'],
            value_dim=config['value_dim'],
            dropout_rate=config['dropout_rate'],
            use_bias=config['use_bias'],
            output_shape=config['output_shape'],
            attention_axes=config['attention_axes'],
            kernel_initializer=config['kernel_initializer'],
            bias_initializer=config['bias_initializer'],
            random_state=config['random_state'],
        )


class PositionalEncoding(Layer):
    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
        warmup_steps: int = 1000,
        initial_scale: float = 0.1,
        final_scale: float = 0.2,
        scale_embeddings: bool = True,
        trainable: bool = False,
        random_state: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.warmup_steps = warmup_steps
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.scale_embeddings = scale_embeddings
        self.trainable = trainable
        self.random_state = random_state

        self.current_step = 0
        self.current_scale = initial_scale

        self.base_scale_factor = 1.0 if not scale_embeddings else 1.0 / \
            np.sqrt(embedding_dim)

        if trainable:
            self.rng = np.random.default_rng(random_state)
            self.initialize_weights()
        else:
            self._build_sinusoidal_encoding()

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _build_sinusoidal_encoding(self) -> None:
        position = np.arange(self.max_sequence_length)[:, np.newaxis]

        div_term = np.power(
            10000.0,
            np.arange(0, self.embedding_dim, 2, dtype=np.float32) /
            self.embedding_dim
        )

        pe = np.zeros((self.max_sequence_length,
                      self.embedding_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.weights = pe[np.newaxis, :, :]
        self.d_weights = np.zeros_like(self.weights)

    def initialize_weights(self) -> None:
        if self.trainable:
            self._build_sinusoidal_encoding()

            noise = self.rng.normal(
                0,
                0.01,
                self.weights.shape
            )
            self.weights = self.weights + noise
            self.d_weights = np.zeros_like(self.weights)

    def get_warmup_scale(self) -> float:
        if self.current_step >= self.warmup_steps:
            return self.final_scale

        progress = self.current_step / self.warmup_steps
        return self.initial_scale + (self.final_scale - self.initial_scale) * progress

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = input_data.shape

        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_sequence_length}")

        input_norm = np.sqrt(
            np.sum(input_data**2, axis=-1, keepdims=True) + 1e-6)
        normalized_input = input_data / input_norm

        pos_encoding = self.weights[:, :seq_len, :]
        if batch_size > 1:
            pos_encoding = np.repeat(pos_encoding, batch_size, axis=0)

        if self.trainable:
            self.current_scale = self.get_warmup_scale()
        else:
            self.current_scale = self.final_scale

        effective_scale = self.current_scale * self.base_scale_factor

        scaled_pe = pos_encoding * effective_scale
        output = normalized_input + scaled_pe

        self.metadata = {
            'step': self.current_step,
            'scale': self.current_scale,
            'effective_scale': effective_scale,
            'embedding_contribution': np.mean(np.abs(scaled_pe)) / np.mean(np.abs(normalized_input))
        }

        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        if self.trainable:
            _, seq_len, _ = output_error.shape

            effective_scale = self.current_scale * self.base_scale_factor
            scaled_error = output_error * effective_scale

            self.d_weights[:, :seq_len,
                           :] += np.sum(scaled_error, axis=0, keepdims=True)

            self.current_step += 1

        return output_error

    def get_config(self) -> dict:
        return {
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'warmup_steps': self.warmup_steps,
            'initial_scale': self.initial_scale,
            'final_scale': self.final_scale,
            'scale_embeddings': self.scale_embeddings,
            'trainable': self.trainable,
            'random_state': self.random_state,
            'current_step': self.current_step,
            'current_scale': self.current_scale
        }

    def __str__(self) -> str:
        return (
            f'PositionalEncodingWithWarmup('
            f'seq_length={self.seq_length}, '
            f'embedding_dim={self.embedding_dim}, '
            f'trainable={self.trainable}, '
            f'warmup_steps={self.warmup_steps}, '
            f'current_step={self.current_step}, '
            f'current_scale={self.current_scale:.4f})'
        )


class FeedForward(Layer):
    def __init__(
        self,
        d_ff: int,
        d_model: int,
        dropout_rate: float = 0.1,
        activation: str = 'gelu',
        kernel_initializer: str = "he_normal",
        output_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        random_state: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_ff: int = d_ff
        self.d_model: int = d_model
        self.dropout_rate: float = dropout_rate
        self.activation_name: str = activation
        self.kernel_initializer: str = kernel_initializer
        self.output_initializer: str = output_initializer
        self.bias_initializer: str = bias_initializer
        self.random_state: int = random_state

        self.dense1 = Dense(
            units=d_ff,
            weights_init=kernel_initializer,
            bias_init=bias_initializer,
            random_state=random_state
        )

        self.dense2 = Dense(
            units=d_model,
            weights_init=output_initializer,
            bias_init=bias_initializer,
            random_state=random_state
        )

        self.activation = Activation.from_name(activation)
        self.dropout = Dropout(dropout_rate, random_state=random_state)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f'FeedForward(d_ff={self.d_ff}, d_model={self.d_model})'

    def forward_pass(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.dense1.forward_pass(input_data)
        x = self.activation.forward_pass(x)

        if training:
            x = self.dropout.forward_pass(x, training=True)

        x = self.dense2.forward_pass(x)
        return x

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        dx = self.dense2.backward_pass(output_error)
        dx = self.dropout.backward_pass(dx)
        dx = self.activation.backward_pass(dx)
        dx = self.dense1.backward_pass(dx)
        return dx

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'd_ff': self.d_ff,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict) -> "FeedForward":
        return FeedForward(
            d_ff=config['d_ff'],
            d_model=config['d_model'],
            dropout_rate=config['dropout_rate'],
            activation=config['activation'],
            kernel_initializer=config['kernel_initializer'],
            bias_initializer=config['bias_initializer'],
            random_state=config['random_state']
        )


class AddNorm(Layer):
    def __init__(
        self,
        epsilon: float = 1e-5,
        gamma_init_std: float = 0.05,
        beta_init_std: float = 0.01,
        grad_clip: float = 1.0,
        grad_scale: float = 0.1,
        warmup_steps: int = 2000,
        random_state: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma_init_std = gamma_init_std
        self.beta_init_std = beta_init_std
        self.grad_clip = grad_clip
        self.grad_scale = grad_scale
        self.warmup_steps = warmup_steps
        self.random_state = random_state

        self.step = 0
        self.gamma = None
        self.beta = None
        self.d_gamma = None
        self.d_beta = None

        self.grad_ema = None
        self.grad_emv = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_weights(self, input_shape: tuple[int, ...]) -> None:
        feature_shape = input_shape[-1]
        rng = np.random.default_rng(self.random_state)

        self.gamma = 1.0 + self.gamma_init_std * \
            rng.normal(0, 1, (1, 1, feature_shape))
        self.beta = self.beta_init_std * \
            rng.normal(0, 1, (1, 1, feature_shape))

        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

        self.grad_ema = np.zeros_like(self.gamma)
        self.grad_emv = np.ones_like(self.gamma)

    def get_warmup_factor(self) -> float:
        return min(1.0, (self.step + 1) / self.warmup_steps)

    def update_gradient_stats(self, grad: np.ndarray) -> None:
        if self.grad_ema is None:
            return

        beta1, beta2 = 0.9, 0.999  # Adam like momentum and variance

        self.grad_ema = beta1 * self.grad_ema + (1 - beta1) * grad

        self.grad_emv = beta2 * self.grad_emv + (1 - beta2) * (grad ** 2)

    def normalize_gradients(self, grad: np.ndarray) -> np.ndarray:
        """Normalize gradients using running statistics"""
        if self.grad_emv is None:
            return grad

        scale = np.sqrt(self.grad_emv) + self.epsilon

        warmup = self.get_warmup_factor()

        normalized = grad / scale * warmup
        return np.clip(normalized, -self.grad_clip, self.grad_clip)

    def forward_pass(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        x, residual = inputs
        self.residual = residual

        combined = x + residual

        if self.gamma is None:
            self.initialize_weights(combined.shape)

        self.mean = np.mean(combined, axis=-1, keepdims=True)
        self.var = np.var(combined, axis=-1, keepdims=True,
                          ddof=1) + self.epsilon

        std = np.sqrt(self.var)
        self.normalized = (combined - self.mean) / std

        self.std = std
        self.output_before_gamma = self.normalized

        return self.gamma * self.normalized + self.beta

    def backward_pass(self, output_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dY = output_error
        B, T, F = dY.shape
        N = F

        x_minus_mean = self.normalized * self.std

        d_gamma = np.sum(dY * self.normalized, axis=(0, 1), keepdims=True)
        d_beta = np.sum(dY, axis=(0, 1), keepdims=True)

        d_normalized = dY * self.gamma

        d_var = np.sum(d_normalized * x_minus_mean * (-0.5) /
                       (self.std**3), axis=-1, keepdims=True)

        d_mean = np.sum(d_normalized * (-1.0 / self.std), axis=-1, keepdims=True) \
            + d_var * np.mean(-2.0 * x_minus_mean, axis=-1, keepdims=True)

        dx = (d_normalized / self.std) + \
            (d_var * 2.0 * x_minus_mean / N) + (d_mean / N)

        self.d_gamma = d_gamma
        self.d_beta = d_beta

        return dx, dx

    def __str__(self) -> str:
        return f'AddNorm(epsilon={self.epsilon}, random_state={self.random_state})'

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'epsilon': self.epsilon,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict) -> "AddNorm":
        return AddNorm(
            epsilon=config['epsilon'],
            random_state=config['random_state']
        )


class TransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = 'gelu',
        kernel_initializer: str = "glorot_normal",
        bias_initializer: str = "zeros",
        random_state: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        key_dim = d_model // num_heads
        self.attention_key_scale = np.sqrt(key_dim)
        self.attention_output_scale = 1.0

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=attention_dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            random_state=random_state
        )

        self.ffn = FeedForward(
            d_ff=d_ff,
            d_model=d_model,
            dropout_rate=dropout_rate,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            random_state=random_state
        )

        self.attention_dropout = Dropout(
            dropout_rate, random_state=random_state)
        self.ffn_dropout = Dropout(dropout_rate, random_state=random_state)

        norm_config = {
            'epsilon': 1e-6,
            'gamma_init_std': 0.02,
            'beta_init_std': 0.01,
            'grad_clip': 1.0,
            'random_state': random_state
        }

        self.attention_norm = AddNorm(**norm_config)
        self.ffn_norm = AddNorm(**norm_config)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward_pass(self, inputs: np.ndarray, mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        self.x = inputs
        attn_output = self.attention.forward_pass(
            self.x, mask=mask, training=training)

        if training:
            attn_output = self.attention_dropout.forward_pass(
                attn_output, training=True)

        attn_output = self.attention_norm.forward_pass((attn_output, self.x))

        ffn_output = self.ffn.forward_pass(attn_output, training=training)

        if training:
            ffn_output = self.ffn_dropout.forward_pass(
                ffn_output, training=True)

        output = self.ffn_norm.forward_pass((ffn_output, attn_output))

        return output

    def backward_pass(self, output_error: np.ndarray) -> np.ndarray:
        ffn_norm_dx, ffn_norm_dresidual = self.ffn_norm.backward_pass(
            output_error)
        ffn_dx = self.ffn_dropout.backward_pass(ffn_norm_dx)

        ffn_dx = self.ffn.backward_pass(ffn_dx)
        ffn_dx = ffn_dx + ffn_norm_dresidual

        attn_norm_dx, attn_norm_dresidual = self.attention_norm.backward_pass(
            ffn_dx)
        attn_dx = self.attention_dropout.backward_pass(attn_norm_dx)

        attn_dx = self.attention.backward_pass(attn_dx)
        dx = attn_dx + attn_norm_dresidual

        return dx

    def __str__(self) -> str:
        return f'TransformerEncoderLayer(d_model={self.d_model}, num_heads={self.num_heads})'

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict) -> "TransformerEncoderLayer":
        return TransformerEncoderLayer(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout_rate=config['dropout_rate'],
            attention_dropout=config['attention_dropout'],
            activation=config['activation'],
            kernel_initializer=config['kernel_initializer'],
            bias_initializer=config['bias_initializer'],
            random_state=config['random_state']
        )


class TransformerDecoderLayer(Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = 'gelu',
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.random_state = random_state

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout_rate=attention_dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            random_state=random_state,
        )

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout_rate=attention_dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            random_state=random_state,
        )

        self.ffn = FeedForward(
            d_ff=d_ff,
            d_model=d_model,
            dropout_rate=dropout_rate,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            random_state=random_state
        )

        self.dropout1 = Dropout(dropout_rate, random_state=random_state)
        self.dropout2 = Dropout(dropout_rate, random_state=random_state)
        self.dropout3 = Dropout(dropout_rate, random_state=random_state)

        self.norm1 = AddNorm(random_state=random_state)
        self.norm2 = AddNorm(random_state=random_state)
        self.norm3 = AddNorm(random_state=random_state)

        self.cache = {}

    def __str__(self) -> str:
        return f'TransformerDecoderLayer(d_model={self.d_model}, num_heads={self.num_heads})'

    def forward_pass(
        self,
        x: np.ndarray,
        enc_output: np.ndarray,
        training: bool = True,
        self_attention_mask: np.ndarray | None = None,
        cross_attention_mask: np.ndarray | None = None
    ) -> np.ndarray:
        self.cache['x'] = x
        self.cache['enc_output'] = enc_output

        # Self attention
        attn1 = self.self_attention.forward_pass(
            x, mask=self_attention_mask, training=training)
        if training:
            attn1 = self.dropout1.forward_pass(attn1, training=True)
        out1 = self.norm1.forward_pass((attn1, x))
        self.cache['attn1'] = attn1
        self.cache['out1'] = out1

        # Cross attention
        attn2 = self.cross_attention.forward_pass(
            (out1, enc_output, enc_output),
            mask=cross_attention_mask,
            training=training
        )

        if training:
            attn2 = self.dropout2.forward_pass(attn2, training=True)
        out2 = self.norm2.forward_pass((attn2, out1))
        self.cache['attn2'] = attn2
        self.cache['out2'] = out2

        # Feed forward
        ffn_out = self.ffn.forward_pass(out2, training=training)
        if training:
            ffn_out = self.dropout3.forward_pass(ffn_out, training=True)
        out3 = self.norm3.forward_pass((ffn_out, out2))
        self.cache['ffn_out'] = ffn_out

        return out3

    def backward_pass(self, output_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d_norm3, d_residual3 = self.norm3.backward_pass(output_error)
        d_ffn = self.dropout3.backward_pass(d_norm3)

        d_ffn = self.ffn.backward_pass(d_ffn)

        d_out2 = d_residual3 + d_ffn

        d_norm2, d_residual2 = self.norm2.backward_pass(d_out2)
        d_attn2 = self.dropout2.backward_pass(d_norm2)

        d_attn2_query, d_attn2_key, d_attn2_value = self.cross_attention.backward_pass(
            d_attn2)
        d_out1 = d_residual2 + normalize_gradient(d_attn2_query)

        d_norm1, d_residual1 = self.norm1.backward_pass(d_out1)
        d_attn1 = self.dropout1.backward_pass(d_norm1)

        d_x = self.self_attention.backward_pass(d_attn1)

        d_x = d_residual1 + d_x

        batch_size, seq_len, hidden_dim = d_attn2_key.shape
        attention_weights = self.cross_attention.attention_weights
        num_heads = self.cross_attention.num_heads
        head_dim = hidden_dim // num_heads

        if attention_weights is None:
            attention_weights = np.ones(
                (batch_size, num_heads, seq_len, seq_len))

        head_importance = np.mean(
            attention_weights, axis=(2, 3), keepdims=True)
        head_importance = head_importance.reshape(batch_size, num_heads, 1, 1)

        d_key_reshaped = d_attn2_key.reshape(
            batch_size * seq_len, num_heads * head_dim)
        d_key_heads = d_key_reshaped.reshape(
            batch_size, seq_len, num_heads, head_dim)
        d_key_heads = np.transpose(d_key_heads, (0, 2, 1, 3))

        d_value_reshaped = d_attn2_value.reshape(
            batch_size * seq_len, num_heads * head_dim)
        d_value_heads = d_value_reshaped.reshape(
            batch_size, seq_len, num_heads, head_dim)
        d_value_heads = np.transpose(d_value_heads, (0, 2, 1, 3))

        d_key_weighted = head_importance * d_key_heads
        d_value_weighted = (1.0 - head_importance) * d_value_heads

        d_combined_heads = d_key_weighted + d_value_weighted

        d_combined = np.transpose(d_combined_heads, (0, 2, 1, 3))
        d_combined = d_combined.reshape(batch_size, seq_len, hidden_dim)

        return d_x, d_combined

    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'activation': self.activation_name,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'random_state': self.random_state
        }

    @staticmethod
    def from_config(config: dict) -> "TransformerDecoderLayer":
        return TransformerDecoderLayer(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout_rate=config['dropout_rate'],
            attention_dropout=config['attention_dropout'],
            activation=config['activation'],
            kernel_initializer=config['kernel_initializer'],
            bias_initializer=config['bias_initializer'],
            random_state=config['random_state']
        )


# --------------------------------------------------------------------------------------------------------------


incompatibility_dict = {
    Input: [],

    Dense: [Conv1D, Conv2D, UpSampling2D],

    Activation: [],

    Conv2D: [Conv1D, LSTM, GRU, Bidirectional, Unidirectional],

    UpSampling2D: [Conv1D, LSTM, GRU, Bidirectional, Unidirectional],
    
    Conv2DTranspose: [Conv1D, LSTM, GRU, Bidirectional, Unidirectional],

    MaxPooling2D: [Conv1D, MaxPooling1D, AveragePooling1D, LSTM, GRU, Bidirectional, Unidirectional],

    AveragePooling2D: [Conv1D, MaxPooling1D, AveragePooling1D, LSTM, GRU, Bidirectional, Unidirectional],

    GlobalAveragePooling2D: [Conv1D, MaxPooling1D, AveragePooling1D, LSTM, GRU, Bidirectional, Unidirectional],

    Conv1D: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D],

    MaxPooling1D: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D],

    AveragePooling1D: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D],

    GlobalAveragePooling1D: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    Flatten: [],

    Dropout: [],

    Embedding: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    BatchNormalization: [],

    LayerNormalization: [],

    Permute: [],

    TextVectorization: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    Reshape: [],

    LSTM: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    GRU: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    Bidirectional: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    Unidirectional: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    Attention: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    MultiHeadAttention: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    PositionalEncoding: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    FeedForward: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    AddNorm: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    TransformerEncoderLayer: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],

    TransformerDecoderLayer: [Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D],
}
