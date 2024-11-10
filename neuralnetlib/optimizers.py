import numpy as np

from neuralnetlib.utils import dict_with_ndarray_to_dict_with_list, dict_with_list_to_dict_with_ndarray


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray,
               bias_grad: np.ndarray):
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}

    @staticmethod
    def from_config(config: dict):
        if config['name'] == 'SGD':
            return SGD.from_config(config)
        elif config['name'] == 'Momentum':
            return Momentum.from_config(config)
        elif config['name'] == 'RMSprop':
            return RMSprop.from_config(config)
        elif config['name'] == 'Adam':
            return Adam.from_config(config)
        else:
            raise ValueError(f"Unknown optimizer name: {config['name']}")

    @staticmethod
    def from_name(name: str) -> "Optimizer":
        name = name.lower().replace("_", "")

        for subclass in Optimizer.__subclasses__():
            if subclass.__name__.lower() == name:
                return subclass()

        raise ValueError(f"No optimizer found for the name: {name}")


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray,
               bias_grad: np.ndarray):
        weights -= self.learning_rate * weights_grad
        bias -= self.learning_rate * bias_grad

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__, "learning_rate": self.learning_rate}

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"

    @staticmethod
    def from_config(config: dict):
        return SGD(config['learning_rate'])


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray,
               bias_grad: np.ndarray):
        if self.velocity is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(bias)
        self.velocity_w = self.momentum * self.velocity_w - \
                          self.learning_rate * weights_grad
        weights += self.velocity_w
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * bias_grad
        bias += self.velocity_b

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__, "learning_rate": self.learning_rate, "momentum": self.momentum,
                "velocity": self.velocity}

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, momentum={self.momentum})"

    @staticmethod
    def from_config(config: dict):
        return Momentum(config['learning_rate'], config['momentum'])


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grads = None

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray,
               bias_grad: np.ndarray):
        if self.sq_grads is None:
            self.sq_grads_w = np.zeros_like(weights)
            self.sq_grads_b = np.zeros_like(bias)
        self.sq_grads_w = self.rho * self.sq_grads_w + \
                          (1 - self.rho) * np.square(weights_grad)
        weights -= self.learning_rate * weights_grad / \
                   (np.sqrt(self.sq_grads_w) + self.epsilon)
        self.sq_grads_b = self.rho * self.sq_grads_b + \
                          (1 - self.rho) * np.square(bias_grad)
        bias -= self.learning_rate * bias_grad / \
                (np.sqrt(self.sq_grads_b) + self.epsilon)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__, "learning_rate": self.learning_rate, "rho": self.rho,
                "epsilon": self.epsilon, "sq_grads": self.sq_grads}

    @staticmethod
    def from_config(config: dict):
        return RMSprop(config['learning_rate'], config['rho'], config['epsilon'])

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, rho={self.rho}, epsilon={self.epsilon})"


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, 
                 epsilon: float = 1e-8, clip_norm: float = None, clip_value: float = None) -> None:

        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.t = 0
        
        self.m_w, self.v_w = {}, {}
        self.m_b, self.v_b = {}, {}
        
        self._min_denom = 1e-16
        
        self._max_exp = np.log(np.finfo(np.float64).max)  # Maximum exponent value for float64 = 709

    def _clip_gradients(self, grad: np.ndarray) -> np.ndarray:
        if grad is None:
            return None
            
        if self.clip_norm is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.clip_norm:
                grad = grad * (self.clip_norm / (grad_norm + self._min_denom))
                
        if self.clip_value is not None:
            grad = np.clip(grad, -self.clip_value, self.clip_value)
            
        return grad

    def _compute_moments(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray) -> tuple:
        grad = self._clip_gradients(grad)
        
        m = self.beta_1 * m + (1 - self.beta_1) * grad
        v = self.beta_2 * v + (1 - self.beta_2) * np.square(grad)
        
        beta1_t = self.beta_1 ** self.t
        beta2_t = self.beta_2 ** self.t
        
        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)
        
        denom = np.sqrt(v_hat) + self.epsilon
        update = self.learning_rate * m_hat / np.maximum(denom, self._min_denom)
        
        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        param -= update
        
        return param, m, v

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray, bias_grad: np.ndarray) -> None:
        if layer_index not in self.m_w:
            self.m_w[layer_index] = np.zeros_like(weights)
            self.v_w[layer_index] = np.zeros_like(weights)
            self.m_b[layer_index] = np.zeros_like(bias)
            self.v_b[layer_index] = np.zeros_like(bias)

        self.t += 1
        
        weights, self.m_w[layer_index], self.v_w[layer_index] = self._compute_moments(
            weights, weights_grad, self.m_w[layer_index], self.v_w[layer_index]
        )
        
        bias, self.m_b[layer_index], self.v_b[layer_index] = self._compute_moments(
            bias, bias_grad, self.m_b[layer_index], self.v_b[layer_index]
        )

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "clip_norm": self.clip_norm,
            "clip_value": self.clip_value,
            "t": self.t,
            "m_w": dict_with_ndarray_to_dict_with_list(self.m_w),
            "v_w": dict_with_ndarray_to_dict_with_list(self.v_w),
            "m_b": dict_with_ndarray_to_dict_with_list(self.m_b),
            "v_b": dict_with_ndarray_to_dict_with_list(self.v_b)
        }

    @staticmethod
    def from_config(config: dict):
        adam = Adam(
            learning_rate=config['learning_rate'],
            beta_1=config['beta_1'],
            beta_2=config['beta_2'],
            epsilon=config['epsilon'],
            clip_norm=config.get('clip_norm'),
            clip_value=config.get('clip_value')
        )
        adam.t = config['t']
        adam.m_w = dict_with_list_to_dict_with_ndarray(config['m_w'])
        adam.v_w = dict_with_list_to_dict_with_ndarray(config['v_w'])
        adam.m_b = dict_with_list_to_dict_with_ndarray(config['m_b'])
        adam.v_b = dict_with_list_to_dict_with_ndarray(config['v_b'])
        return adam

    def __str__(self):
        return (f"{self.__class__.__name__}(learning_rate={self.learning_rate}, "
                f"beta_1={self.beta_1}, beta_2={self.beta_2}, epsilon={self.epsilon}, "
                f"clip_norm={self.clip_norm}, clip_value={self.clip_value})")