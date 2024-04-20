import unittest

import numpy as np

from neuralnetlib.activations import Sigmoid, ReLU, Tanh, Softmax, Linear, LeakyReLU, ELU, SELU


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        sigmoid = Sigmoid()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid(x), expected_output)

    def test_relu(self):
        relu = ReLU()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.maximum(0, x)
        np.testing.assert_array_almost_equal(relu(x), expected_output)

    def test_tanh(self):
        tanh = Tanh()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.tanh(x)
        np.testing.assert_array_almost_equal(tanh(x), expected_output)

    def test_softmax(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        expected_output = exps / np.sum(exps, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(softmax(x), expected_output)

    def test_linear(self):
        linear = Linear()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = x
        np.testing.assert_array_almost_equal(linear(x), expected_output)

    def test_leaky_relu(self):
        leaky_relu = LeakyReLU()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.where(x > 0, x, x * 0.01)
        np.testing.assert_array_almost_equal(leaky_relu(x), expected_output)

    def test_elu(self):
        elu = ELU()
        x = np.array([-1.0, 0.0, 1.0])
        expected_output = np.where(x > 0, x, np.exp(x) - 1)
        np.testing.assert_array_almost_equal(elu(x), expected_output)

    def test_selu(self):
        selu = SELU()
        x = np.array([-1.0, 0.0, 1.0])
        alpha = SELU.DEFAULT_ALPHA
        scale = SELU.DEFAULT_SCALE
        expected_output = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        np.testing.assert_array_almost_equal(selu(x), expected_output)

    def test_softmax_derivative_not_implemented(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(NotImplementedError):
            softmax.derivative(x)


if __name__ == '__main__':
    unittest.main()
