import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        name = config.get('name')
        if not name:
            raise ValueError('Config must contain "name" field')

        constructor_params = {k: v for k, v in config.items() 
                            if k not in ['name', 'config']}

        for activation_class in ActivationFunction.__subclasses__():
            if activation_class.__name__ == name:
                return activation_class(**constructor_params)

        raise ValueError(f'Unknown activation function: {name}')


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        clipped_x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-clipped_x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        activated_x = self(x)
        return activated_x * (1 - activated_x)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(self(x))

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Derivative of Softmax is not implemented. It is not needed for backpropagation.")

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}


class Linear(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, x * self.alpha)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            'alpha': self.alpha
        }

    def __str__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class ELU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, np.exp(x) - 1)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, np.exp(x))

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}

    def __str__(self):
        return self.__class__.__name__


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
            "name": self.__class__.__name__,
            'alpha': self.alpha,
            'scale': self.scale
        }

    def __str__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, scale={self.scale})"


class GELU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        z = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        tanh_z = np.tanh(z)
        g = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        return 0.5 * (1 + tanh_z + x * (1 - tanh_z**2) * g)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}

    def __str__(self):
        return self.__class__.__name__
