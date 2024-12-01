import numpy as np


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
                distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
                probs = distances / distances.sum()
                cumprobs = np.cumsum(probs)
                r = self.rng.random()
                ind = np.searchsorted(cumprobs, r)
                centroids.append(X[ind])
                
            return np.array(centroids)
            
        elif self.init == 'random':
            indices = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
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
                inertia += np.sum((X[labels == k] - self.cluster_centers_[k]) ** 2)
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
            
            centroid_shift = np.sum((new_centroids - self.cluster_centers_) ** 2)
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
