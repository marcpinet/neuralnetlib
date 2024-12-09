import numpy as np


class LossFunction:
    EPSILON = 1e-15

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}

    @staticmethod
    def from_config(config: dict) -> 'LossFunction':
        loss_name = config['name']

        for loss_class in LossFunction.__subclasses__():
            if loss_class.__name__ == loss_name:
                constructor_params = {k: v for k,
                                      v in config.items() if k != 'name'}
                return loss_class(**constructor_params)

    @staticmethod
    def from_name(name: str) -> "LossFunction":
        aliases = {
            "mse": "MeanSquaredError",
            "bce": "BinaryCrossentropy",
            "cce": "CategoricalCrossentropy",
            "scce": "SparseCategoricalCrossentropy",
            "mae": "MeanAbsoluteError",
            "kld": "KullbackLeiblerDivergence",
            "cels": "CrossEntropyWithLabelSmoothing",
            "wass": "Wasserstein",
            "focal": "FocalLoss",
            "fl": "FocalLoss",
            "bfocal": "BinaryFocalLossPerLabel",
            "bfl": "BinaryFocalLossPerLabel",
            "asymmetric": "AsymmetricLoss",
            "multibce": "MultiLabelBCELoss"
        }

        original_name = name
        name = name.lower().replace("_", "")

        if name.startswith("huber") and len(original_name.split("_")) == 2:
            try:
                delta = float(original_name.split("_")[-1])
                return Huber(delta=delta)
            except ValueError:
                pass

        if name in aliases:
            name = aliases[name]
            
        for loss_class in LossFunction.__subclasses__():
            if loss_class.__name__.lower() == name or loss_class.__name__ == name:
                if loss_class.__name__ == "Huber":
                    return loss_class(delta=1.0)
                elif loss_class.__name__ == "CrossEntropyWithLabelSmoothing":
                    return loss_class(label_smoothing=0.1)
                elif loss_class.__name__ == "FocalLoss":
                    return loss_class(gamma=2.0, alpha=0.25)
                elif loss_class.__name__ == "AsymmetricLoss":
                    return loss_class(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
                elif loss_class.__name__ == "BinaryFocalLossPerLabel":
                    return loss_class(gamma=2.0, alpha=0.25)
                elif loss_class.__name__ == "MultiLabelBCELoss":
                    return loss_class(pos_weight=1.0)
                else:
                    return loss_class()

        raise ValueError(
            f"No loss function found for the name: {original_name}")


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]

    def __str__(self):
        return "MeanSquaredError"


class BinaryCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON,
                         1 - LossFunction.EPSILON)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, LossFunction.EPSILON,
                         1 - LossFunction.EPSILON)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def __str__(self):
        return "BinaryCrossentropy"


class CategoricalCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON,
                         1 - LossFunction.EPSILON)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        try:
            y_pred = np.clip(y_pred, LossFunction.EPSILON,
                             1 - LossFunction.EPSILON)
            return -y_true / y_pred
        except Exception as e:
            print(e, "Make sure to one-hot encode your labels.", sep="\n")

    def __str__(self):
        return "CategoricalCrossentropy"


class SparseCategoricalCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON,
                         1 - LossFunction.EPSILON)

        batch_size = y_true.shape[0]
        y_pred_selected = y_pred[np.arange(batch_size), y_true]

        return -np.mean(np.log(y_pred_selected))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, LossFunction.EPSILON,
                         1 - LossFunction.EPSILON)

        batch_size = y_true.shape[0]
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(batch_size), y_true] = 1

        return -y_true_one_hot / y_pred

    def __str__(self):
        return "SparseCategoricalCrossentropy"


class MeanAbsoluteError(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.where(y_pred > y_true, 1, -1)

    def __str__(self):
        return "MeanAbsoluteError"


class Huber(LossFunction):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_true - y_pred
        return np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error))

    def __str__(self):
        return f"HuberLoss(delta={self.delta})"

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__, "delta": self.delta}


class KullbackLeiblerDivergence(LossFunction):
    def __call__(self, mu: np.ndarray, log_var: np.ndarray) -> float:
        return -0.5 * np.mean(1 + log_var - np.square(mu) - np.exp(log_var))

    def derivative(self, mu: np.ndarray, log_var: np.ndarray) -> tuple:
        d_mu = mu
        d_log_var = 0.5 * (np.exp(log_var) - 1)
        return d_mu, d_log_var

    def __str__(self):
        return "KullbackLeiblerDivergence"


class CrossEntropyWithLabelSmoothing(LossFunction):
    def __init__(self, label_smoothing: float = 0.1):
        self.label_smoothing = label_smoothing
        self.epsilon = 1e-15

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32),
                         self.epsilon, 1 - self.epsilon)

        if y_pred.ndim != 3 or y_true.ndim != 2 or y_true.shape != y_pred.shape[:2]:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        n_classes = y_pred.shape[-1]
        batch_size, seq_length = y_true.shape

        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(batch_size)[:, None],
                np.arange(seq_length), y_true] = 1.0

        smooth_one_hot = (1.0 - self.label_smoothing) * \
            one_hot + self.label_smoothing / n_classes

        loss = -np.sum(smooth_one_hot * np.log(y_pred)) / \
            (batch_size * seq_length)
        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32),
                         self.epsilon, 1 - self.epsilon)

        if y_pred.ndim != 3 or y_true.ndim != 2 or y_true.shape != y_pred.shape[:2]:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        n_classes = y_pred.shape[-1]
        batch_size, seq_length = y_true.shape

        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(batch_size)[:, None],
                np.arange(seq_length), y_true] = 1.0

        smooth_one_hot = (1.0 - self.label_smoothing) * \
            one_hot + self.label_smoothing / n_classes

        grad = -smooth_one_hot * \
            (1.0 / (y_pred + self.epsilon)) / (batch_size * seq_length)
        return grad

    def __str__(self):
        return f"CrossEntropyWithLabelSmoothing(label_smoothing={self.label_smoothing})"


class Wasserstein(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true * y_pred)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true

    def __str__(self):
        return "Wasserstein"


class FocalLoss(LossFunction):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)

        ce_loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = np.power(1 - p_t, self.gamma)

        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        focal_loss = alpha_factor * modulating_factor * ce_loss

        return np.mean(focal_loss)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        modulating_factor = np.power(1 - p_t, self.gamma)
        d_modulating_factor = -self.gamma * np.power(1 - p_t, self.gamma - 1)

        d_ce = y_true / y_pred - (1 - y_true) / (1 - y_pred)

        derivative = alpha_factor * (
            modulating_factor * d_ce +
            d_modulating_factor *
            (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        )

        return derivative / y_true.shape[0]

    def __str__(self):
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha})"

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "gamma": self.gamma,
            "alpha": self.alpha
        }


class BinaryFocalLossPerLabel(LossFunction):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, scale: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.scale = scale
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        bce = y_true * -np.log(y_pred) + (1 - y_true) * -np.log(1 - y_pred)
        bce = np.clip(bce, -100, 100)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = np.clip(p_t, self.EPSILON, 1 - self.EPSILON)
        focusing_factor = np.power(1 - p_t, self.gamma)
        focusing_factor = np.clip(focusing_factor, 0, 100)
        
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focusing_factor * bce * self.scale
        
        return np.mean(focal_loss)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        p_t = np.clip(p_t, self.EPSILON, 1 - self.EPSILON)
        
        alpha_factor = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        
        focusing_factor = np.power(1 - p_t, self.gamma - 1)
        focusing_factor = np.clip(focusing_factor, 0, 100)
        
        sign = np.where(y_true == 1, -1.0, 1.0)
        
        modulating = (self.gamma * p_t * np.log(p_t + self.EPSILON) + 1)
        modulating = np.clip(modulating, -100, 100)
        
        grad = (sign * alpha_factor * focusing_factor * modulating) * self.scale
        
        grad = grad / (y_true.shape[0] * y_true.shape[1])
        grad = np.clip(grad, -10, 10)
        
        return grad

class MultiLabelBCELoss(LossFunction):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        weights = np.where(y_true == 1, self.pos_weight, 1.0)
        
        bce = -(weights * y_true * np.log(y_pred) + 
                (1 - y_true) * np.log(1 - y_pred))
        
        return np.mean(bce)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        weights = np.where(y_true == 1, self.pos_weight, 1.0)
        
        grad = weights * (y_pred - y_true) / (y_pred * (1 - y_pred) + self.EPSILON)
        grad = np.clip(grad, -10, 10)
        
        return grad / y_true.size


class AsymmetricLoss(LossFunction):
    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        if self.clip > 0:
            y_pred = np.clip(y_pred, self.clip, 1 - self.clip)
            
        pos_mask = (y_true == 1)
        xs_pos = np.where(pos_mask, y_pred, 1)
        xs_neg = np.where(~pos_mask, 1 - y_pred, 1)
        
        pos_focusing = np.where(pos_mask, np.power(1 - xs_pos, self.gamma_pos), 1)
        neg_focusing = np.where(~pos_mask, np.power(1 - xs_neg, self.gamma_neg), 1)
        
        pos_bce = np.where(pos_mask, -np.log(xs_pos), 0)
        neg_bce = np.where(~pos_mask, -np.log(xs_neg), 0)
        
        loss = pos_focusing * pos_bce + neg_focusing * neg_bce
        return np.mean(loss)
        
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        
        if self.clip > 0:
            y_pred = np.clip(y_pred, self.clip, 1 - self.clip)
            
        pos_mask = (y_true == 1)
        
        d_pos_focusing = self.gamma_pos * np.power(1 - y_pred, self.gamma_pos - 1)
        d_neg_focusing = self.gamma_neg * np.power(y_pred, self.gamma_neg - 1)
        
        d_pos_bce = -1 / y_pred
        d_neg_bce = 1 / (1 - y_pred)
        
        gradient = np.where(pos_mask,
            d_pos_focusing * d_pos_bce,
            d_neg_focusing * d_neg_bce
        )
        
        return gradient / y_true.shape[0]