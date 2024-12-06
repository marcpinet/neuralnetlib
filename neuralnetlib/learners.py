import numpy as np


class IsolationTree:
    def __init__(self, height_limit: int, random_state: int = None):
        self.height_limit = height_limit
        self.size = 0
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, current_height: int = 0):
        self.size = X.shape[0]

        if current_height >= self.height_limit or self.size <= 1:
            return

        X = np.atleast_2d(X)
        feature_min = np.min(X, axis=0)
        feature_max = np.max(X, axis=0)

        features_range = feature_max - feature_min
        valid_features = np.nonzero(features_range > 0)[0]

        if len(valid_features) == 0:
            return

        self.split_feature = self.rng.choice(valid_features)
        min_val = feature_min[self.split_feature]
        max_val = feature_max[self.split_feature]
        self.split_value = self.rng.uniform(min_val, max_val)

        left_indices = X[:, self.split_feature] < self.split_value
        X_left = X[left_indices]
        X_right = X[~left_indices]

        if X_left.shape[0] > 0:
            self.left = IsolationTree(self.height_limit)
            self.left.fit(X_left, current_height + 1)

        if X_right.shape[0] > 0:
            self.right = IsolationTree(self.height_limit)
            self.right.fit(X_right, current_height + 1)

    def path_length(self, x: np.ndarray, current_height: int = 0) -> float:
        if self.left is None and self.right is None:
            return current_height

        x = np.atleast_1d(x)
        if x[self.split_feature] < self.split_value:
            if self.left is None:
                return current_height
            return self.left.path_length(x, current_height + 1)
        else:
            if self.right is None:
                return current_height
            return self.right.path_length(x, current_height + 1)


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, contamination: float = 0.1, random_state: int = None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.height_limit = int(np.ceil(np.log2(max_samples)))
        self.trees = []
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray) -> 'IsolationForest':
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        self.trees = []
        for _ in range(self.n_estimators):
            indices = self.rng.integers(
                0, n_samples, min(self.max_samples, n_samples))
            X_sample = X[indices]
            tree = IsolationTree(self.height_limit)
            tree.fit(X_sample)
            self.trees.append(tree)

        self.c = self._average_path_length(self.max_samples)
        scores = self.score_samples(X)
        self.threshold = np.percentile(scores, 100 * self.contamination)

        return self

    def _average_path_length(self, n: int) -> float:
        if n <= 1:
            return 1
        precision = 1000
        terms = np.arange(1, precision + 1)
        euler_mascheroni = np.sum(1 / terms) - np.log(precision)
        return 2 * (np.log(n - 1) + euler_mascheroni) - 2 * (n - 1) / n

    def path_length(self, x: np.ndarray) -> float:
        return np.mean([tree.path_length(x) for tree in self.trees])

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = X.reshape(-1, 1)

        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            h = self.path_length(x)
            scores[i] = -2 ** (-h / self.c)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)


class DecisionTree:
    def __init__(self, tree_type: str = "classifier", max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: int = None, random_state: int = None):
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_classes = None
        self.rng = np.random.default_rng(random_state)

    class Node:
        def __init__(self):
            self.left = None
            self.right = None
            self.feature = None
            self.threshold = None
            self.prediction = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y, features):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        current_metric = self._gini(
            y) if self.tree_type == "classifier" else self._mse(y)

        for feature in features:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                if self.tree_type == "classifier":
                    left_metric = self._gini(y[left_mask])
                    right_metric = self._gini(y[right_mask])
                else:
                    left_metric = self._mse(y[left_mask])
                    right_metric = self._mse(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = n_left + n_right

                gain = current_metric - \
                    (n_left/n_total * left_metric + n_right/n_total * right_metric)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        node = self.Node()

        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf:
            if self.tree_type == "classifier":
                unique, counts = np.unique(y, return_counts=True)
                node.prediction = unique[np.argmax(counts)]
            else:
                node.prediction = np.mean(y)
            return node

        n_features_to_consider = self.max_features or X.shape[1]
        features = self.rng.choice(
            X.shape[1], size=n_features_to_consider, replace=False)

        feature, threshold = self._best_split(X, y, features)

        if feature is None:
            if self.tree_type == "classifier":
                unique, counts = np.unique(y, return_counts=True)
                node.prediction = unique[np.argmax(counts)]
            else:
                node.prediction = np.mean(y)
            return node

        left_mask = X[:, feature] <= threshold

        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _predict_single(self, x, node):
        if node.prediction is not None:
            return node.prediction

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def fit(self, X, y):
        if self.tree_type == "classifier":
            self.n_classes = len(np.unique(y))
            y = y.astype(int)

        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        if self.tree_type == "classifier":
            return predictions.astype(int)
        return predictions


class RandomForest:
    def __init__(self, n_estimators: int = 100, tree_type: str = "classifier",
                 max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str | int = "sqrt", bootstrap: bool = True,
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        return n_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        if self.tree_type == "classifier":
            y = y.astype(int)
            self.classes_ = np.unique(y)

        base_seed = self.random_state if self.random_state is not None else self.rng.integers(
            0, 1000000)

        for i in range(self.n_estimators):
            tree = DecisionTree(
                tree_type=self.tree_type,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=base_seed + i if base_seed is not None else None
            )

            if self.bootstrap:
                indices = self.rng.choice(
                    n_samples, size=n_samples, replace=True)
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)

            self.trees.append(tree)

        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.tree_type == "classifier":
            mode_predictions = []
            for sample_pred in predictions.T:
                values, counts = np.unique(
                    sample_pred.astype(int), return_counts=True)
                mode_predictions.append(values[np.argmax(counts)])
            return np.array(mode_predictions)
        return np.mean(predictions, axis=0)


class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column <= self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.stumps = []
        self.rng = np.random.default_rng(random_state)

    def _build_stump(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        best_stump = DecisionStump()

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            if len(thresholds) == 1:
                continue

            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                polarity_error = np.zeros(2)
                predictions = np.ones(n_samples)

                # 1
                predictions[feature_values <= threshold] = -1
                polarity_error[0] = np.sum(weights[predictions != y])

                # -1
                predictions = np.ones(n_samples)
                predictions[feature_values > threshold] = -1
                polarity_error[1] = np.sum(weights[predictions != y])

                error = np.min(polarity_error)
                if error < min_error:
                    min_error = error
                    best_stump.feature_idx = feature_idx
                    best_stump.threshold = threshold
                    best_stump.polarity = 1 if polarity_error[0] < polarity_error[1] else -1

        eps = 1e-10
        best_stump.alpha = 0.5 * \
            np.log((1.0 - min_error + eps) / (min_error + eps))
        return best_stump, min_error

    def fit(self, X, y):
        n_samples = X.shape[0]

        y = np.where(y <= 0, -1, 1)
        weights = np.ones(n_samples) / n_samples

        self.stumps = []

        for _ in range(self.n_estimators):
            stump, error = self._build_stump(X, y, weights)

            if error > 0.5:
                break

            predictions = stump.predict(X)
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)

            if error < 1e-10:
                break

        return self

    def predict(self, X):
        return np.sign(self.score_samples(X))

    def predict_proba(self, X):
        scores = self.score_samples(X)
        proba = 1 / (1 + np.exp(-2 * scores))
        return np.vstack([1 - proba, proba]).T

    def score_samples(self, X):
        return np.sum([stump.alpha * stump.predict(X) for stump in self.stumps], axis=0)


class DecisionTreeGBM:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.value = None
            self.left = None
            self.right = None

    def _best_split(self, X, y):
        m = X.shape[0]
        if m < self.min_samples_split:
            return None, None, None

        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        S = np.sum((y - np.mean(y)) ** 2)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:
                continue

            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                gain = S - (
                    np.sum((y[left_mask] - np.mean(y[left_mask])) ** 2) +
                    np.sum((y[right_mask] - np.mean(y[right_mask])) ** 2)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        node = self.Node()

        if depth >= self.max_depth:
            node.value = np.mean(y)
            return node

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            node.value = np.mean(y)
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])


class GradientBoostingMachine:
    def __init__(self, task="regression", n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2, subsample=1.0, random_state=None):
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.rng = np.random.default_rng(random_state)

        self.trees = []
        self.initial_prediction = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_residuals(self, y_true, y_pred):
        if self.task == "regression":
            return y_true - y_pred
        else:
            p = self._sigmoid(y_pred)
            return y_true - p

    def _sample_indices(self, n_samples):
        if self.subsample == 1:
            return np.arange(n_samples)
        n_subsamples = int(n_samples * self.subsample)
        return self.rng.choice(n_samples, size=n_subsamples, replace=False)

    def fit(self, X, y):
        n_samples = X.shape[0]

        if self.task == "regression":
            self.initial_prediction = np.mean(y)
        else:
            y = np.where(y <= 0, 0, 1)
            self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))

        F = np.full(n_samples, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = self._compute_residuals(y, F)
            indices = self._sample_indices(n_samples)

            tree = DecisionTreeGBM(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X[indices], residuals[indices])
            self.trees.append(tree)

            predictions = tree.predict(X)
            F += self.learning_rate * predictions

        return self

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        if self.task == "binary_classification":
            return (self._sigmoid(predictions) >= 0.5).astype(int)
        return predictions

    def predict_proba(self, X):
        if self.task != "binary_classification":
            raise ValueError(
                "predict_proba is only available for binary classification")

        predictions = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        proba = self._sigmoid(predictions)
        return np.vstack([1 - proba, proba]).T


class XGBoostNode:
    def __init__(self):
        self.feature_idx: int = None
        self.threshold: float = None
        self.left: 'XGBoostNode' = None
        self.right: 'XGBoostNode' = None
        self.value: float = None
        self.gain: float = 0.0
        self.cover: float = 0.0


class XGBoostTree:
    def __init__(self, max_depth: int = 6, min_child_weight: float = 1.0,
                 lambda_: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma
        self.root = None

    def _calc_leaf_value(self, grad: np.ndarray, hess: np.ndarray) -> float:
        return -np.sum(grad) / (np.sum(hess) + self.lambda_)

    def _calc_gain(self, grad: np.ndarray, hess: np.ndarray) -> float:
        G, H = np.sum(grad), np.sum(hess)
        return (G * G) / (H + self.lambda_)

    def _find_best_split(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> tuple:
        best_gain = 0.0
        best_feature_idx = None
        best_threshold = None
        total_gain = self._calc_gain(grad, hess)
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if (np.sum(hess[left_mask]) < self.min_child_weight or
                        np.sum(hess[right_mask]) < self.min_child_weight):
                    continue

                left_gain = self._calc_gain(grad[left_mask], hess[left_mask])
                right_gain = self._calc_gain(
                    grad[right_mask], hess[right_mask])
                gain = left_gain + right_gain - total_gain - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray,
                    depth: int = 0) -> XGBoostNode:
        node = XGBoostNode()
        node.cover = np.sum(hess)

        if depth >= self.max_depth:
            node.value = self._calc_leaf_value(grad, hess)
            return node

        feature_idx, threshold, gain = self._find_best_split(X, grad, hess)

        if feature_idx is None or gain <= 0:
            node.value = self._calc_leaf_value(grad, hess)
            return node

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.gain = gain
        node.left = self._build_tree(X[left_mask], grad[left_mask],
                                     hess[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], grad[right_mask],
                                      hess[right_mask], depth + 1)

        return node

    def _predict_sample(self, x: np.ndarray, node: XGBoostNode) -> float:
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def fit(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> 'XGBoostTree':
        self.root = self._build_tree(X, grad, hess)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(x, self.root) for x in X])


class XGBoost:
    def __init__(self, objective: str = "reg:squarederror", n_estimators: int = 100,
                 learning_rate: float = 0.3, max_depth: int = 6,
                 min_child_weight: float = 1.0, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, lambda_: float = 1.0,
                 gamma: float = 0.0, random_state: int = None):
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambda_ = lambda_
        self.gamma = gamma
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.trees = []
        self.base_score = 0.5

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _compute_gradients(self, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.objective == "reg:squarederror":
            grad = pred - y
            hess = np.ones_like(y)
        else:
            prob = self._sigmoid(pred)
            grad = prob - y
            hess = prob * (1 - prob)
        return grad, hess

    def _subsample_data(self, X: np.ndarray, y: np.ndarray,
                        grad: np.ndarray, hess: np.ndarray) -> tuple:
        # Row subsampling
        if self.subsample < 1.0:
            n_samples = int(X.shape[0] * self.subsample)
            indices = self.rng.choice(
                X.shape[0], size=n_samples, replace=False)
            X = X[indices]
            y = y[indices]
            grad = grad[indices]
            hess = hess[indices]

        if self.colsample_bytree < 1.0:
            n_features = int(X.shape[1] * self.colsample_bytree)
            feature_indices = self.rng.choice(
                X.shape[1], size=n_features, replace=False)
            X = X[:, feature_indices]

        return X, y, grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoost':
        if self.objective == "binary:logistic":
            y = (y > 0).astype(np.float64)
            self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-6))
        else:
            self.base_score = np.mean(y)

        predictions = np.full(X.shape[0], self.base_score)
        self.trees = []

        for _ in range(self.n_estimators):
            grad, hess = self._compute_gradients(y, predictions)

            X_tree, y_tree, grad_tree, hess_tree = self._subsample_data(
                X, y, grad, hess)

            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                lambda_=self.lambda_,
                gamma=self.gamma
            )
            tree.fit(X_tree, grad_tree, hess_tree)
            self.trees.append(tree)

            predictions += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        if self.objective == "binary:logistic":
            return (self._sigmoid(predictions) >= 0.5).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.objective != "binary:logistic":
            raise ValueError(
                "predict_proba is only available for binary classification")

        predictions = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        proba = self._sigmoid(predictions)
        return np.vstack([1 - proba, proba]).T


class SVM:
    def __init__(self,
                 learning_rate: float = 0.01,
                 lambda_param: float = 0.01,
                 n_iters: int = 1000,
                 random_state: int = None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.rng = np.random.default_rng(random_state)

    def _initialize_weights(self, X: np.ndarray):
        n_features = X.shape[1]
        self.w = self.rng.standard_normal(n_features)
        self.b = 0

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples = X.shape[0]
        distances = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - distances)
        cost = np.sum(hinge_loss) / n_samples

        l2_reg = self.lambda_param * np.sum(self.w ** 2)
        return cost + l2_reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        y = np.where(y <= 0, -1, 1)

        self._initialize_weights(X)
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.w) + self.b

            condition = y * linear_output < 1

            dw = (self.lambda_param * 2 * self.w -
                  np.dot(X[condition].T, y[condition])) / n_samples
            db = -np.sum(y[condition]) / n_samples

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.w) + self.b
        y_pred = np.sign(linear_output)
        return np.where(y_pred <= 0, 0, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        proba = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - proba, proba]).T


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init='kmeans++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.rng = np.random.default_rng(random_state)

        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0
        self.inertia_ = None

    def _init_centroids(self, X):
        n_samples = X.shape[0]

        if self.init == 'kmeans++':
            centroids = [X[self.rng.integers(n_samples)]]

            for _ in range(1, self.n_clusters):
                distances = np.min([np.sum((X - c) ** 2, axis=1)
                                   for c in centroids], axis=0)
                probs = distances / distances.sum()
                cumprobs = np.cumsum(probs)
                r = self.rng.random()
                ind = np.searchsorted(cumprobs, r)
                centroids.append(X[ind])

            return np.array(centroids)

        elif self.init == 'random':
            indices = self.rng.choice(
                n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        else:
            raise ValueError("init must be 'kmeans++' or 'random'")

    def _find_nearest_cluster(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.cluster_centers_):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(X[labels == k], axis=0)
            else:
                new_centroids[k] = X[self.rng.integers(X.shape[0])]
        return new_centroids

    def _compute_inertia(self, X, labels):
        inertia = 0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                inertia += np.sum((X[labels == k] -
                                  self.cluster_centers_[k]) ** 2)
        return inertia

    def fit_predict(self, X):
        return self.fit(X).labels_

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.cluster_centers_ = self._init_centroids(X)
        prev_labels = None

        for i in range(self.max_iter):
            labels = self._find_nearest_cluster(X)

            if prev_labels is not None and np.all(labels == prev_labels):
                break

            new_centroids = self._compute_centroids(X, labels)

            centroid_shift = np.sum(
                (new_centroids - self.cluster_centers_) ** 2)
            if centroid_shift < self.tol:
                break

            self.cluster_centers_ = new_centroids
            prev_labels = labels
            self.n_iter_ = i + 1

        self.labels_ = self._find_nearest_cluster(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._find_nearest_cluster(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.cluster_centers_):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        return distances


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_clusters_ = 0

    def _get_neighbors(self, X, sample_idx):
        if self.metric == 'euclidean':
            distances = np.sum((X - X[sample_idx]) ** 2, axis=1)
            return np.nonzero(distances <= self.eps ** 2)[0]
        else:
            raise ValueError("Only euclidean metric is supported")

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)

        visited = np.zeros(n_samples, dtype=bool)
        core_samples = np.zeros(n_samples, dtype=bool)

        cluster_label = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue

            core_samples[i] = True
            self.labels_[i] = cluster_label

            neighbors = list(neighbors)
            j = 0
            while j < len(neighbors):
                neighbor = neighbors[j]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    new_neighbors = self._get_neighbors(X, neighbor)

                    if len(new_neighbors) >= self.min_samples:
                        core_samples[neighbor] = True
                        neighbors.extend(set(new_neighbors) - set(neighbors))

                if self.labels_[neighbor] == -1:
                    self.labels_[neighbor] = cluster_label

                j += 1

            cluster_label += 1

        self.core_sample_indices_ = np.nonzero(core_samples)[0]
        self.components_ = X[core_samples]
        self.n_clusters_ = cluster_label

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
