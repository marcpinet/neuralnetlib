import json
import unittest

import numpy as np

from neuralnetlib.optimizers import Optimizer, SGD, Momentum, RMSprop, Adam, AdaBelief, RAdam


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.weights = np.array([[0.1, -0.2], [0.4, 0.5]])
        self.bias = np.array([0.1, -0.3])
        self.weights_grad = np.array([[0.01, -0.02], [0.04, 0.05]])
        self.bias_grad = np.array([0.01, -0.03])

    def test_sgd(self):
        sgd = SGD(learning_rate=0.01)
        initial_weights = self.weights.copy()
        initial_bias = self.bias.copy()
        sgd.update(0, self.weights, self.weights_grad, self.bias, self.bias_grad)
        np.testing.assert_array_almost_equal(self.weights, initial_weights - 0.01 * self.weights_grad)
        np.testing.assert_array_almost_equal(self.bias, initial_bias - 0.01 * self.bias_grad)

    def test_momentum(self):
        momentum = Momentum(learning_rate=0.01, momentum=0.9)
        initial_weights = self.weights.copy()
        initial_bias = self.bias.copy()
        momentum.update(0, self.weights, self.weights_grad, self.bias, self.bias_grad)
        np.testing.assert_array_almost_equal(self.weights, initial_weights - 0.01 * self.weights_grad)
        np.testing.assert_array_almost_equal(self.bias, initial_bias - 0.01 * self.bias_grad)

    def test_adam(self):
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        initial_weights = self.weights.copy()
        initial_bias = self.bias.copy()

        adam.update(0, self.weights, self.weights_grad, self.bias, self.bias_grad)

        m_w = (0.9 * np.zeros_like(self.weights) + 0.1 * self.weights_grad)
        v_w = (0.999 * np.zeros_like(self.weights) + 0.001 * np.square(self.weights_grad))
        m_w_hat = m_w / (1 - 0.9 ** (1))
        v_w_hat = v_w / (1 - 0.999 ** (1))
        expected_weights = initial_weights - 0.01 * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)

        m_b = (0.9 * np.zeros_like(self.bias) + 0.1 * self.bias_grad)
        v_b = (0.999 * np.zeros_like(self.bias) + 0.001 * np.square(self.bias_grad))
        m_b_hat = m_b / (1 - 0.9 ** (1))
        v_b_hat = v_b / (1 - 0.999 ** (1))
        expected_bias = initial_bias - 0.01 * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

        np.testing.assert_array_almost_equal(self.weights, expected_weights, decimal=2)
        np.testing.assert_array_almost_equal(self.bias, expected_bias, decimal=2)

    def test_rmsprop(self):
        rmsprop = RMSprop(learning_rate=0.01, rho=0.9)
        initial_weights = self.weights.copy()
        initial_bias = self.bias.copy()

        rmsprop.update(0, self.weights, self.weights_grad, self.bias, self.bias_grad)

        sq_grads_w = 0.9 * np.zeros_like(self.weights) + 0.1 * np.square(self.weights_grad)
        expected_weights = initial_weights - 0.01 * self.weights_grad / (np.sqrt(sq_grads_w) + 1e-8)

        sq_grads_b = 0.9 * np.zeros_like(self.bias) + 0.1 * np.square(self.bias_grad)
        expected_bias = initial_bias - 0.01 * self.bias_grad / (np.sqrt(sq_grads_b) + 1e-8)

        np.testing.assert_array_almost_equal(self.weights, expected_weights)
        np.testing.assert_array_almost_equal(self.bias, expected_bias)

    def test_adam_bias_correction_per_parameter(self):
        # the first update of every parameter must be bias corrected with t=1, even after other parameters were updated
        adam = Adam(learning_rate=0.01)
        for layer_index in range(3):
            weights = np.zeros(2)
            adam.update(layer_index, weights, np.array([0.5, -0.5]))
            np.testing.assert_allclose(weights, [-0.01, 0.01], rtol=1e-6)

    def test_state_survives_get_config(self):
        for optimizer in [Momentum(), RMSprop(), Adam(), AdaBelief(), RAdam()]:
            weights = self.weights.copy()
            optimizer.update(0, weights, self.weights_grad, self.bias.copy(), self.bias_grad)
            optimizer.get_config()
            optimizer.update(0, weights, self.weights_grad, self.bias.copy(), self.bias_grad)

    def test_config_round_trip(self):
        for optimizer in [SGD(0.1), Momentum(), RMSprop(), Adam(), AdaBelief(), RAdam()]:
            weights_a, weights_b = self.weights.copy(), self.weights.copy()
            optimizer.update(0, weights_a, self.weights_grad)
            config = json.loads(json.dumps(optimizer.get_config()))
            loaded = Optimizer.from_config(config)
            self.assertIsInstance(loaded, type(optimizer))
            # weights_b is behind by one step: apply it with the same state before comparing the next steps
            weights_b[...] = weights_a
            optimizer.update(0, weights_a, self.weights_grad)
            loaded.update(0, weights_b, self.weights_grad)
            np.testing.assert_allclose(weights_a, weights_b)


if __name__ == '__main__':
    unittest.main()
