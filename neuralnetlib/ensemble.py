import numpy as np
from enum import Enum


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
            indices = self.rng.integers(0, n_samples, min(self.max_samples, n_samples))
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


class TreeType(Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class DecisionTree:
    def __init__(self, tree_type: TreeType = TreeType.CLASSIFIER, max_depth: int = None, 
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
        
        current_metric = self._gini(y) if self.tree_type == TreeType.CLASSIFIER else self._mse(y)
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                if self.tree_type == TreeType.CLASSIFIER:
                    left_metric = self._gini(y[left_mask])
                    right_metric = self._gini(y[right_mask])
                else:
                    left_metric = self._mse(y[left_mask])
                    right_metric = self._mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = n_left + n_right
                
                gain = current_metric - (n_left/n_total * left_metric + n_right/n_total * right_metric)
                
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
            if self.tree_type == TreeType.CLASSIFIER:
                unique, counts = np.unique(y, return_counts=True)
                node.prediction = unique[np.argmax(counts)]
            else:
                node.prediction = np.mean(y)
            return node
        
        n_features_to_consider = self.max_features or X.shape[1]
        features = self.rng.choice(X.shape[1], size=n_features_to_consider, replace=False)
        
        feature, threshold = self._best_split(X, y, features)
        
        if feature is None:
            if self.tree_type == TreeType.CLASSIFIER:
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
        if self.tree_type == TreeType.CLASSIFIER:
            self.n_classes = len(np.unique(y))
            y = y.astype(int)
            
        self.root = self._build_tree(X, y)
        return self
        
    def predict(self, X):
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        if self.tree_type == TreeType.CLASSIFIER:
            return predictions.astype(int)
        return predictions


class RandomForest:
    def __init__(self, n_estimators: int = 100, tree_type: TreeType = TreeType.CLASSIFIER, 
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
        
        if self.tree_type == TreeType.CLASSIFIER:
            y = y.astype(int)
            self.classes_ = np.unique(y)
        
        base_seed = self.random_state if self.random_state is not None else self.rng.integers(0, 1000000)
        
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
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                tree.fit(X[indices], y[indices])
            else:
                tree.fit(X, y)
                
            self.trees.append(tree)
            
        return self
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.tree_type == TreeType.CLASSIFIER:
            mode_predictions = []
            for sample_pred in predictions.T:
                values, counts = np.unique(sample_pred.astype(int), return_counts=True)
                mode_predictions.append(values[np.argmax(counts)])
            return np.array(mode_predictions)
        return np.mean(predictions, axis=0)


import numpy as np

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
        best_stump.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))
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


import numpy as np
from enum import Enum

class GBMTask(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"

class DecisionTree:
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
    def __init__(self, task=GBMTask.REGRESSION, n_estimators=100, learning_rate=0.1, 
                 max_depth=3, min_samples_split=2, subsample=1.0, random_state=None):
        self.task = task if isinstance(task, GBMTask) else GBMTask(task)
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
        if self.task == GBMTask.REGRESSION:
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
        
        if self.task == GBMTask.REGRESSION:
            self.initial_prediction = np.mean(y)
        else:
            y = np.where(y <= 0, 0, 1)
            self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))
            
        F = np.full(n_samples, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = self._compute_residuals(y, F)
            indices = self._sample_indices(n_samples)
            
            tree = DecisionTree(
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
            
        if self.task == GBMTask.BINARY_CLASSIFICATION:
            return (self._sigmoid(predictions) >= 0.5).astype(int)
        return predictions
        
    def predict_proba(self, X):
        if self.task != GBMTask.BINARY_CLASSIFICATION:
            raise ValueError("predict_proba is only available for binary classification")
            
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        proba = self._sigmoid(predictions)
        return np.vstack([1 - proba, proba]).T
