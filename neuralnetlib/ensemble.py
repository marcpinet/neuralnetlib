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
