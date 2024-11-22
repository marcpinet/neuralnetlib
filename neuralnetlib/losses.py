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
        if config['name'] == 'MeanSquaredError':
            return MeanSquaredError()
        elif config['name'] == 'BinaryCrossentropy':
            return BinaryCrossentropy()
        elif config['name'] == 'CategoricalCrossentropy':
            return CategoricalCrossentropy()
        elif config['name'] == 'MeanAbsoluteError':
            return MeanAbsoluteError()
        elif config['name'] == 'HuberLoss':
            return HuberLoss(config['delta'])
        elif config['name'] == 'KullbackLeiblerDivergence':
            return KullbackLeiblerDivergence()
        else:
            raise ValueError(f'Unknown loss function: {config["name"]}')

    @staticmethod
    def from_name(name: str) -> "LossFunction":
        name = name.lower().replace("_", "")
        if name == "mse" or name == "meansquarederror":
            return MeanSquaredError()
        elif name == "bce" or name == "binarycrossentropy":
            return BinaryCrossentropy()
        elif name == "cce" or name == "categorycrossentropy":
            return CategoricalCrossentropy()
        elif name == "scce" or name == "sparsecategoricalcrossentropy":
            return SparseCategoricalCrossentropy()
        elif name == "mae" or name == "meanabsoluteerror":
            return MeanAbsoluteError()
        elif name == "kld" or name == "kullbackleiblerdivergence":
            return KullbackLeiblerDivergence()
        elif name == "sequencecrossentropy" or name == "sce":
            return SequenceCrossEntropy()
        elif name == "crossentropywithlabelsmoothing" or name == "cels":
            return CrossEntropyWithLabelSmoothing()
        elif name.startswith("huber") and len(name.split("_")) == 2:
            delta = float(name.split("_")[-1])
            return HuberLoss(delta)
        else:
            for subclass in LossFunction.__subclasses__():
                if subclass.__name__.lower() == name:
                    return subclass()

        raise ValueError(f"No loss function found for the name: {name}")


class MeanSquaredError(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]

    def __str__(self):
        return "MeanSquaredError"


class BinaryCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def __str__(self):
        return "BinaryCrossentropy"


class CategoricalCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        try:
            y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)
            return -y_true / y_pred
        except Exception as e:
            print(e, "Make sure to one-hot encode your labels.", sep="\n")

    def __str__(self):
        return "CategoricalCrossentropy"


class SparseCategoricalCrossentropy(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)

        batch_size = y_true.shape[0]
        y_pred_selected = y_pred[np.arange(batch_size), y_true]

        return -np.mean(np.log(y_pred_selected))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, LossFunction.EPSILON, 1 - LossFunction.EPSILON)

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


class HuberLoss(LossFunction):
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
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), self.epsilon, 1 - self.epsilon)

        if y_pred.ndim != 3 or y_true.ndim != 2 or y_true.shape != y_pred.shape[:2]:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        n_classes = y_pred.shape[-1]
        batch_size, seq_length = y_true.shape

        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(batch_size)[:, None], np.arange(seq_length), y_true] = 1.0

        smooth_one_hot = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing / n_classes

        loss = -np.sum(smooth_one_hot * np.log(y_pred)) / (batch_size * seq_length)
        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), self.epsilon, 1 - self.epsilon)

        if y_pred.ndim != 3 or y_true.ndim != 2 or y_true.shape != y_pred.shape[:2]:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        n_classes = y_pred.shape[-1]
        batch_size, seq_length = y_true.shape

        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(batch_size)[:, None], np.arange(seq_length), y_true] = 1.0

        smooth_one_hot = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing / n_classes

        grad = -(smooth_one_hot / y_pred) / (batch_size * seq_length)
        return grad

    def __str__(self):
        return f"CrossEntropyWithLabelSmoothing(label_smoothing={self.label_smoothing})"
