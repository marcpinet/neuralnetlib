import unittest

import numpy as np

from neuralnetlib.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy, MeanAbsoluteError, \
    Huber


class TestLossFunctions(unittest.TestCase):

    def test_mean_squared_error(self):
        mse = MeanSquaredError()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        expected_loss = np.mean(np.power(y_true - y_pred, 2))
        calculated_loss = mse(y_true, y_pred)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_binary_crossentropy(self):
        bce = BinaryCrossentropy()
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0.1, 0.8, 0.99])
        expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        calculated_loss = bce(y_true, y_pred)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_categorical_crossentropy(self):
        cce = CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        expected_loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        calculated_loss = cce(y_true, y_pred)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_mean_absolute_error(self):
        mae = MeanAbsoluteError()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        expected_loss = np.mean(np.abs(y_true - y_pred))
        calculated_loss = mae(y_true, y_pred)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_huber_loss(self):
        huber = Huber(delta=1.0)
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        error = y_true - y_pred
        is_small_error = np.abs(error) <= huber.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = huber.delta * (np.abs(error) - 0.5 * huber.delta)
        expected_loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))
        calculated_loss = huber(y_true, y_pred)
        self.assertAlmostEqual(calculated_loss, expected_loss)


if __name__ == '__main__':
    unittest.main()
