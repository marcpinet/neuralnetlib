import unittest

import numpy as np

from neuralnetlib.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


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


if __name__ == '__main__':
    unittest.main()
