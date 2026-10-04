import unittest

import numpy as np

from neuralnetlib.learners import IsolationForest, RandomForest, AdaBoost, GradientBoostingMachine, XGBoost, SVM, \
    KMeans
from neuralnetlib.utils import make_blobs, make_classification


class TestLearners(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X, self.y = make_classification(n_samples=150, n_features=2, n_classes=2, n_clusters_per_class=1,
                                             n_informative=2, n_redundant=0, random_state=42)

    def test_isolation_forest(self):
        data = np.vstack([self.rng.normal(0, 1, (100, 2)), self.rng.uniform(6, 9, (5, 2))])
        first = IsolationForest(n_estimators=30, max_samples=64, random_state=0).fit(data).predict(data)
        second = IsolationForest(n_estimators=30, max_samples=64, random_state=0).fit(data).predict(data)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[100:] == -1))

    def test_refit_replaces_trees(self):
        forest = RandomForest(n_estimators=3, random_state=0)
        forest.fit(self.X, self.y)
        forest.fit(self.X, self.y)
        self.assertEqual(len(forest.trees), 3)

        gbm = GradientBoostingMachine(task="binary_classification", n_estimators=5)
        gbm.fit(self.X, self.y)
        gbm.fit(self.X, self.y)
        self.assertEqual(len(gbm.trees), 5)

    def test_classifiers_keep_the_labels(self):
        for labels in [self.y, self.y + 1, np.where(self.y == 1, 1, -1)]:
            for model in [AdaBoost(n_estimators=10), SVM(n_iters=300, random_state=0)]:
                predictions = model.fit(self.X, labels).predict(self.X)
                self.assertTrue(set(np.unique(predictions)) <= set(np.unique(labels)), type(model).__name__)
                self.assertGreater(np.mean(predictions == labels), 0.8, type(model).__name__)

    def test_xgboost_column_subsampling(self):
        X = self.rng.uniform(-3, 3, (150, 3))
        y = 2 * X[:, 0] - X[:, 2]
        model = XGBoost(n_estimators=40, learning_rate=0.2, max_depth=3, colsample_bytree=0.67, random_state=0)
        model.fit(X, y)
        self.assertLess(np.mean((model.predict(X) - y) ** 2), 0.5)

    def test_kmeans(self):
        X, y = make_blobs(n_samples=90, centers=3, random_state=1, cluster_std=0.5)
        labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
        self.assertEqual(len(np.unique(labels)), 3)
        KMeans(n_clusters=2, random_state=0).fit(np.ones((5, 2)))


if __name__ == '__main__':
    unittest.main()
