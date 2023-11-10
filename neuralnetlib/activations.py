import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def from_config(config: dict):
        name = config['name']
        if name == 'Sigmoid':
            return Sigmoid()
        elif name == 'ReLU':
            return ReLU()
        elif name == 'Tanh':
            return Tanh()
        elif name == 'Softmax':
            return Softmax()
        elif name == 'Linear':
            return Linear()
        elif name == 'LeakyReLU':
            return LeakyReLU()
        elif name == 'ELU':
            return ELU()
        elif name == 'SELU':
            return SELU(alpha=config['alpha'], scale=config['scale'])
        else:
            raise ValueError(f'Unknown activation function: {name}')


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        clipped_x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-clipped_x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        activated_x = self(x)
        return activated_x * (1 - activated_x)


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(self(x))


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Derivative of Softmax is not implemented. It is not needed for backpropagation.")


class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class LeakyReLU(ActivationFunction):
    LEAKY_SLOPE = 0.01

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, x * LeakyReLU.LEAKY_SLOPE)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, LeakyReLU.LEAKY_SLOPE)


class ELU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, np.exp(x) - 1)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, np.exp(x))


class SELU(ActivationFunction):
    # See: https://arxiv.org/pdf/1706.02515.pdf and https://pytorch.org/docs/stable/generated/torch.nn.SELU.html for more details
    DEFAULT_ALPHA = 1.6732632423543772848170429916717
    DEFAULT_SCALE = 1.0507009873554804934193349852946

    def __init__(self, alpha: float = None, scale: float = None):
        self.alpha = alpha if alpha is not None else SELU.DEFAULT_ALPHA
        self.scale = scale if scale is not None else SELU.DEFAULT_SCALE

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x > 0, 1, self.alpha * np.exp(x))

    def get_config(self) -> dict:
        return {
            'alpha': self.alpha,
            'scale': self.scale
        }
