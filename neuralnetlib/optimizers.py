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
        optimizer_name = config['name']

        for optimizer_class in Optimizer.__subclasses__():
            if optimizer_class.__name__ == optimizer_name:
                constructor_params = {k: v for k,
                                      v in config.items() if k != 'name'}
                return optimizer_class(**constructor_params)

        raise ValueError(f"No optimizer found for the name: {optimizer_name}")

    @staticmethod
    def from_name(name: str) -> "Optimizer":
        name = name.lower().replace("_", "")

        for subclass in Optimizer.__subclasses__():
            if subclass.__name__.lower() == name:
                return subclass()

        raise ValueError(f"No optimizer found for the name: {name}")


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        super().__init__(learning_rate)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray = None, bias_grad: np.ndarray = None):
        weights -= self.learning_rate * weights_grad
        if bias is not None and bias_grad is not None:
            bias -= self.learning_rate * bias_grad

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__, "learning_rate": self.learning_rate}

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"

    @staticmethod
    def from_config(config: dict):
        return SGD(config['learning_rate'])


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, **kwargs):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = {}
        self.velocity_b = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, layer_index, weights, weights_grad, bias=None, bias_grad=None):
        if layer_index not in self.velocity_w:
            self.velocity_w[layer_index] = np.zeros_like(weights)
        if bias is not None and layer_index not in self.velocity_b:
            self.velocity_b[layer_index] = np.zeros_like(bias)

        self.velocity_w[layer_index] = (
            self.momentum * self.velocity_w[layer_index] - self.learning_rate * weights_grad
        )
        weights += self.velocity_w[layer_index]

        if bias is not None:
            self.velocity_b[layer_index] = (
                self.momentum * self.velocity_b[layer_index] - self.learning_rate * bias_grad
            )
            bias += self.velocity_b[layer_index]

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "velocity_w": dict_with_ndarray_to_dict_with_list(self.velocity_w) if hasattr(self, 'velocity_w') else None,
            "velocity_b": dict_with_ndarray_to_dict_with_list(self.velocity_b) if hasattr(self, 'velocity_b') else None
        }

    @staticmethod
    def from_config(config: dict):
        optimizer = Momentum(config['learning_rate'], config['momentum'])
        if config.get('velocity_w'):
            optimizer.velocity_w = dict_with_list_to_dict_with_ndarray(config['velocity_w'])
            optimizer.velocity_b = dict_with_list_to_dict_with_ndarray(config['velocity_b'])
        return optimizer

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, momentum={self.momentum})"


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9, epsilon: float = 1e-8, **kwargs):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grads_w = {}
        self.sq_grads_b = {}
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, layer_index, weights, weights_grad, bias=None, bias_grad=None):
        if layer_index not in self.sq_grads_w:
            self.sq_grads_w[layer_index] = np.zeros_like(weights)
        if bias is not None and layer_index not in self.sq_grads_b:
            self.sq_grads_b[layer_index] = np.zeros_like(bias)

        self.sq_grads_w[layer_index] = (
            self.rho * self.sq_grads_w[layer_index] + (1 - self.rho) * np.square(weights_grad)
        )
        weights -= (
            self.learning_rate * weights_grad
            / (np.sqrt(self.sq_grads_w[layer_index]) + self.epsilon)
        )

        if bias is not None:
            self.sq_grads_b[layer_index] = (
                self.rho * self.sq_grads_b[layer_index] + (1 - self.rho) * np.square(bias_grad)
            )
            bias -= (
                self.learning_rate * bias_grad
                / (np.sqrt(self.sq_grads_b[layer_index]) + self.epsilon)
            )

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "sq_grads_w": dict_with_ndarray_to_dict_with_list(self.sq_grads_w) if hasattr(self, 'sq_grads_w') else None,
            "sq_grads_b": dict_with_ndarray_to_dict_with_list(self.sq_grads_b) if hasattr(self, 'sq_grads_b') else None
        }

    @staticmethod
    def from_config(config: dict):
        optimizer = RMSprop(config['learning_rate'], config['rho'], config['epsilon'])
        if config.get('sq_grads_w'):
            optimizer.sq_grads_w = dict_with_list_to_dict_with_ndarray(config['sq_grads_w'])
            optimizer.sq_grads_b = dict_with_list_to_dict_with_ndarray(config['sq_grads_b'])
        return optimizer

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, rho={self.rho}, epsilon={self.epsilon})"


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8, clip_norm: float = None, clip_value: float = None, **kwargs) -> None:

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

        # Maximum exponent value for float64 = 709
        self._max_exp = np.log(np.finfo(np.float64).max)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        if param is None or grad is None:
            return None, None, None

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

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, 
              bias: np.ndarray = None, bias_grad: np.ndarray = None) -> None:
        if layer_index not in self.m_w:
            self.m_w[layer_index] = np.zeros_like(weights)
            self.v_w[layer_index] = np.zeros_like(weights)
            if bias is not None:
                self.m_b[layer_index] = np.zeros_like(bias)
                self.v_b[layer_index] = np.zeros_like(bias)

        self.t += 1

        weights, self.m_w[layer_index], self.v_w[layer_index] = self._compute_moments(
            weights, weights_grad, self.m_w[layer_index], self.v_w[layer_index]
        )

        if bias is not None and bias_grad is not None:
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


class AdaBelief(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-16, clip_norm: float = None, clip_value: float = None, **kwargs) -> None:
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.t = 0

        self.m_w, self.s_w = {}, {}
        self.m_b, self.s_b = {}, {}

        self._min_denom = 1e-16
        self._max_exp = np.log(np.finfo(np.float64).max)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

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

    def _compute_moments(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, s: np.ndarray) -> tuple:
        grad = self._clip_gradients(grad)

        m = self.beta_1 * m + (1 - self.beta_1) * grad

        grad_residual = grad - m

        s = self.beta_2 * s + (1 - self.beta_2) * np.square(grad_residual)

        beta1_t = self.beta_1 ** self.t
        beta2_t = self.beta_2 ** self.t

        m_hat = m / (1 - beta1_t)
        s_hat = s / (1 - beta2_t)

        denom = np.sqrt(s_hat + self.epsilon)
        update = self.learning_rate * m_hat / \
            np.maximum(denom, self._min_denom)

        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        param -= update

        return param, m, s

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray = None, bias_grad: np.ndarray = None) -> None:
        if layer_index not in self.m_w:
            self.m_w[layer_index] = np.zeros_like(weights)
            self.s_w[layer_index] = np.zeros_like(weights)
            if bias is not None:
                self.m_b[layer_index] = np.zeros_like(bias)
                self.s_b[layer_index] = np.zeros_like(bias)

        self.t += 1

        weights, self.m_w[layer_index], self.s_w[layer_index] = self._compute_moments(
            weights, weights_grad, self.m_w[layer_index], self.s_w[layer_index]
        )

        if bias is not None and bias_grad is not None:
            bias, self.m_b[layer_index], self.s_b[layer_index] = self._compute_moments(
                bias, bias_grad, self.m_b[layer_index], self.s_b[layer_index]
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
            "s_w": dict_with_ndarray_to_dict_with_list(self.s_w),
            "m_b": dict_with_ndarray_to_dict_with_list(self.m_b),
            "s_b": dict_with_ndarray_to_dict_with_list(self.s_b)
        }

    @staticmethod
    def from_config(config: dict):
        adabelief = AdaBelief(
            learning_rate=config['learning_rate'],
            beta_1=config['beta_1'],
            beta_2=config['beta_2'],
            epsilon=config['epsilon'],
            clip_norm=config.get('clip_norm'),
            clip_value=config.get('clip_value')
        )
        adabelief.t = config['t']
        adabelief.m_w = dict_with_list_to_dict_with_ndarray(config['m_w'])
        adabelief.s_w = dict_with_list_to_dict_with_ndarray(config['s_w'])
        adabelief.m_b = dict_with_list_to_dict_with_ndarray(config['m_b'])
        adabelief.s_b = dict_with_list_to_dict_with_ndarray(config['s_b'])
        return adabelief

    def __str__(self):
        """Retourne une représentation string de l'optimiseur."""
        return (f"{self.__class__.__name__}(learning_rate={self.learning_rate}, "
                f"beta_1={self.beta_1}, beta_2={self.beta_2}, epsilon={self.epsilon}, "
                f"clip_norm={self.clip_norm}, clip_value={self.clip_value})")


class RAdam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8, clip_norm: float = None, clip_value: float = None, **kwargs) -> None:
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
        self._max_exp = np.log(np.finfo(np.float64).max)

        self.rho_inf = 2/(1-beta_2) - 1
        
        for key, value in kwargs.items():
            setattr(self, key, value)

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

        rho_t = self.rho_inf - 2 * self.t * beta2_t / (1 - beta2_t)

        if rho_t > 4:
            v_hat = np.sqrt(v / (1 - beta2_t))
            r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * self.rho_inf) /
                          ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t))

            denom = v_hat + self.epsilon
            update = r_t * self.learning_rate * m_hat / \
                np.maximum(denom, self._min_denom)
        else:
            update = self.learning_rate * m_hat

        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        param -= update

        return param, m, v

    def update(self, layer_index: int, weights: np.ndarray, weights_grad: np.ndarray, bias: np.ndarray = None, bias_grad: np.ndarray = None) -> None:
        if layer_index not in self.m_w:
            self.m_w[layer_index] = np.zeros_like(weights)
            self.v_w[layer_index] = np.zeros_like(weights)
            if bias is not None:
                self.m_b[layer_index] = np.zeros_like(bias)
                self.v_b[layer_index] = np.zeros_like(bias)

        self.t += 1

        weights, self.m_w[layer_index], self.v_w[layer_index] = self._compute_moments(
            weights, weights_grad, self.m_w[layer_index], self.v_w[layer_index]
        )

        if bias is not None and bias_grad is not None:
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
        radam = RAdam(
            learning_rate=config['learning_rate'],
            beta_1=config['beta_1'],
            beta_2=config['beta_2'],
            epsilon=config['epsilon'],
            clip_norm=config.get('clip_norm'),
            clip_value=config.get('clip_value')
        )
        radam.t = config['t']
        radam.m_w = dict_with_list_to_dict_with_ndarray(config['m_w'])
        radam.v_w = dict_with_list_to_dict_with_ndarray(config['v_w'])
        radam.m_b = dict_with_list_to_dict_with_ndarray(config['m_b'])
        radam.v_b = dict_with_list_to_dict_with_ndarray(config['v_b'])
        return radam

    def __str__(self):
        return (f"{self.__class__.__name__}(learning_rate={self.learning_rate}, "
                f"beta_1={self.beta_1}, beta_2={self.beta_2}, epsilon={self.epsilon}, "
                f"clip_norm={self.clip_norm}, clip_value={self.clip_value})")
