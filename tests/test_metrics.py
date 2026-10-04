import unittest

import numpy as np

from neuralnetlib.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, Metric, roc_auc_score, \
    skew, kurtosis, pearsonr, adjusted_rand_score, adjusted_mutual_info_score, jaccard_similarity, precision_at_k


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    def test_accuracy_score(self):
        expected_accuracy = 1.0
        calculated_accuracy = accuracy_score(self.y_pred, self.y_true)
        self.assertAlmostEqual(calculated_accuracy, expected_accuracy)

    def test_f1_score(self):
        expected_f1 = 1.0
        calculated_f1 = f1_score(self.y_pred, self.y_true)
        self.assertAlmostEqual(calculated_f1, expected_f1)

    def test_recall_score(self):
        expected_recall = 1.0
        calculated_recall = recall_score(self.y_pred, self.y_true)
        self.assertAlmostEqual(calculated_recall, expected_recall)

    def test_confusion_matrix(self):
        expected_confusion_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        calculated_confusion_matrix = confusion_matrix(self.y_pred, self.y_true)
        self.assertTrue(np.array_equal(calculated_confusion_matrix, expected_confusion_matrix))

    def test_confusion_matrix_missing_class(self):
        # class 2 never appears: the matrix must still cover the classes 0 to 3
        y_true = np.eye(4)[[0, 1, 3, 1]]
        y_pred = np.eye(4)[[0, 1, 3, 3]]
        cm = confusion_matrix(y_pred, y_true)
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(cm[1, 3], 1)

    def test_accuracy_with_sparse_labels(self):
        y_pred = np.eye(3)[[0, 1, 2, 1]]
        self.assertAlmostEqual(accuracy_score(y_pred, np.array([0, 1, 2, 2])), 0.75)
        self.assertAlmostEqual(accuracy_score(y_pred, np.array([[0], [1], [2], [2]])), 0.75)
        # sequences of predictions
        self.assertAlmostEqual(accuracy_score(np.eye(3)[[[0, 1], [2, 1]]], np.array([[0, 1], [2, 2]])), 0.75)

    def test_metric_wrapper(self):
        self.assertAlmostEqual(Metric('r2')(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])), 1.0)
        self.assertEqual(Metric(Metric('accuracy')).name, 'accuracy')
        # different distributions must give a positive discrepancy (and not NaN)
        self.assertGreater(Metric('mmd')(np.zeros((10, 2)), np.ones((10, 2))), 0.5)
        self.assertAlmostEqual(Metric('mmd')(np.ones((10, 2)), np.ones((10, 2))), 0.0)

    def test_roc_auc(self):
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc_score(scores, labels), 0.75)

    def test_statistics(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        deviations = x - x.mean()
        m2, m3, m4 = np.mean(deviations ** 2), np.mean(deviations ** 3), np.mean(deviations ** 4)
        self.assertAlmostEqual(skew(x), m3 / m2 ** 1.5)
        self.assertAlmostEqual(kurtosis(x), m4 / m2 ** 2 - 3)

        # reference values (scipy.stats.pearsonr)
        r, p_value = pearsonr(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 1.0, 4.0, 3.0, 5.0]))
        self.assertAlmostEqual(float(r), 0.8)
        self.assertAlmostEqual(float(p_value), 0.10408803866182788, places=10)

    def test_clustering_scores(self):
        labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 0, 2])
        # reference values (sklearn.metrics)
        self.assertAlmostEqual(adjusted_rand_score(labels_pred, labels_true), 0.4318181818181818, places=10)
        self.assertAlmostEqual(adjusted_mutual_info_score(labels_pred, labels_true), 0.47728999000145694, places=8)
        self.assertAlmostEqual(adjusted_mutual_info_score(labels_true, labels_true), 1.0)

    def test_multilabel_metrics(self):
        y_pred = np.array([[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]])
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        self.assertAlmostEqual(jaccard_similarity(y_pred, y_true), (1 / 2 + 1 / 2) / 2)
        self.assertAlmostEqual(precision_at_k(y_pred, y_true, k=1), 1.0)


if __name__ == '__main__':
    unittest.main()
