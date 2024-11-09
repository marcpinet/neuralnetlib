import numpy as np
from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def __call__(self, weights: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        pass
    
    def get_config(self) -> dict:
        return {'name': self.__class__.__name__}
        
    @staticmethod
    def from_config(config: dict) -> 'Regularizer':
        regularizer_map = {
            'L1': L1,
            'L2': L2,
            'L1L2': L1L2,
            'OrthogonalRegularizer': OrthogonalRegularizer
        }
        name = config.pop('name')
        return regularizer_map[name](**config)


class L1(Regularizer):
    def __init__(self, l1: float = 0.01):
        self.l1 = l1
        
    def __call__(self, weights: np.ndarray) -> float:
        return self.l1 * np.sum(np.abs(weights))
        
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l1 * np.sign(weights)
        
    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'l1': self.l1
        }


class L2(Regularizer):
    def __init__(self, l2: float = 0.01):
        self.l2 = l2
        
    def __call__(self, weights: np.ndarray) -> float:
        return self.l2 * np.sum(np.square(weights))
        
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2 * self.l2 * weights
        
    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'l2': self.l2
        }


class L1L2(Regularizer):
    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        self.l1 = l1
        self.l2 = l2
        
    def __call__(self, weights: np.ndarray) -> float:
        return (self.l1 * np.sum(np.abs(weights)) + 
                self.l2 * np.sum(np.square(weights)))
        
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l1 * np.sign(weights) + 2 * self.l2 * weights
        
    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'l1': self.l1,
            'l2': self.l2
        }


class OrthogonalRegularizer(Regularizer):
    def __init__(self, factor: float = 0.01):
        self.factor = factor
        
    def __call__(self, weights: np.ndarray) -> float:
        if len(weights.shape) < 2:
            return 0.
        
        original_shape = weights.shape
        if len(original_shape) > 2:
            weights = weights.reshape(-1, original_shape[-1])
            
        rows, cols = weights.shape
        if rows < cols:
            weights = weights.T
            
        identity = np.eye(weights.shape[1])
        weight_product = np.dot(weights.T, weights)
        return self.factor * np.sum(np.square(weight_product - identity))
        
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        if len(weights.shape) < 2:
            return np.zeros_like(weights)
            
        original_shape = weights.shape
        if len(original_shape) > 2:
            weights = weights.reshape(-1, original_shape[-1])
            
        rows, cols = weights.shape
        transposed = False
        if rows < cols:
            weights = weights.T
            transposed = True
            
        identity = np.eye(weights.shape[1])
        weight_product = np.dot(weights.T, weights)
        grad = 4 * self.factor * np.dot(weights, weight_product - identity)
        
        if transposed:
            grad = grad.T
            
        if len(original_shape) > 2:
            grad = grad.reshape(original_shape)
            
        return grad
        
    def get_config(self) -> dict:
        return {
            'name': self.__class__.__name__,
            'factor': self.factor
        }


class AdaptiveDropout:
    def __init__(self, 
                 initial_rate: float = 0.5,
                 min_rate: float = 0.1,
                 max_rate: float = 0.9,
                 temperature: float = 1.0,
                 random_state: int = None):
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.temperature = temperature
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.current_rate = initial_rate
        self.mask = None
        
    def __call__(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return input_data
            
        input_variance = np.var(input_data)
        input_mean = np.mean(np.abs(input_data))
        
        dropout_rate = self.initial_rate * np.exp(-input_variance / (self.temperature * input_mean + 1e-8))
        self.current_rate = np.clip(dropout_rate, self.min_rate, self.max_rate)
        
        self.mask = self.rng.binomial(1, 1 - self.current_rate, 
                                    size=input_data.shape) / (1 - self.current_rate)
        
        return input_data * self.mask
        
    def gradient(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.mask
        
    def get_config(self) -> dict:
        return {
            'initial_rate': self.initial_rate,
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,
            'temperature': self.temperature,
            'random_state': self.random_state
        }
        
    @staticmethod
    def from_config(config: dict) -> 'AdaptiveDropout':
        return AdaptiveDropout(**config)