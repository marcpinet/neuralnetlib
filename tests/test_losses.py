import unittest

import numpy as np

from neuralnetlib.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy, MeanAbsoluteError, \
    Huber, SparseCategoricalCrossentropy, FocalLoss, BinaryFocalLossPerLabel, AsymmetricLoss, MultiLabelBCELoss, \
    KullbackLeiblerDivergence, CrossEntropyWithLabelSmoothing, Wasserstein, LossFunction


def numerical_gradient(f, x, eps=1e-7):
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        old_value = x[index]
        x[index] = old_value + eps
        f_plus = f()
        x[index] = old_value - eps
        f_minus = f()
        x[index] = old_value
        grad[index] = (f_plus - f_minus) / (2 * eps)
    return grad


class TestLossDerivatives(unittest.TestCase):
    """The derivatives must be the gradients of the losses (finite differences)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y_true = (rng.random((6, 3)) > 0.5).astype(float)
        self.y_pred = rng.uniform(0.1, 0.9, (6, 3))
        self.targets = rng.standard_normal((6, 3))

    def assert_derivative(self, loss, y_true, y_pred, scale=1.0):
        expected = numerical_gradient(lambda: loss(y_true, y_pred), y_pred)
        np.testing.assert_allclose(loss.derivative(y_true, y_pred), expected * scale, rtol=1e-4, atol=1e-8)

    def test_regression_losses(self):
        self.assert_derivative(MeanSquaredError(), self.targets, self.y_pred.copy())
        self.assert_derivative(MeanAbsoluteError(), self.targets, self.y_pred.copy())
        self.assert_derivative(Huber(delta=0.5), self.targets, self.y_pred.copy())
        self.assert_derivative(Wasserstein(), np.sign(self.targets), self.y_pred.copy())

    def test_focal_losses(self):
        self.assert_derivative(FocalLoss(gamma=2.0, alpha=0.25), self.y_true, self.y_pred.copy())
        self.assert_derivative(BinaryFocalLossPerLabel(gamma=2.0, alpha=0.25), self.y_true, self.y_pred.copy())
        self.assert_derivative(AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05), self.y_true, self.y_pred.copy())
        self.assert_derivative(MultiLabelBCELoss(pos_weight=2.0), self.y_true, self.y_pred.copy(), scale=1 / 1)

    def test_cross_entropies_are_not_averaged_derivatives(self):
        # (the batch averaging of BCE/CCE is left to the models, which combine them with sigmoid/softmax)
        y_pred = self.y_pred.copy()
        self.assert_derivative(BinaryCrossentropy(), self.y_true, y_pred, scale=self.y_true.size)
        y_true = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        self.assert_derivative(CategoricalCrossentropy(), y_true, y_pred, scale=y_true.shape[0])

    def test_kl_divergence(self):
        kld = KullbackLeiblerDivergence()
        mu = np.random.default_rng(1).standard_normal((4, 3))
        log_var = np.random.default_rng(2).standard_normal((4, 3))
        d_mu, d_log_var = kld.derivative(mu, log_var)
        np.testing.assert_allclose(d_mu, numerical_gradient(lambda: kld(mu, log_var), mu), rtol=1e-5)
        np.testing.assert_allclose(d_log_var, numerical_gradient(lambda: kld(mu, log_var), log_var), rtol=1e-5)

    def test_label_smoothing_cross_entropy(self):
        loss = CrossEntropyWithLabelSmoothing(label_smoothing=0.1)
        rng = np.random.default_rng(3)
        y_true = np.array([[1, 2, 0], [3, 0, 0]])
        y_pred = rng.dirichlet(np.ones(4), size=(2, 3))
        self.assert_derivative(loss, y_true, y_pred)

    def test_sparse_categorical_crossentropy_label_shapes(self):
        scce = SparseCategoricalCrossentropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        labels = np.array([0, 1, 2])
        expected = -np.mean(np.log([0.7, 0.8, 0.6]))
        self.assertAlmostEqual(scce(labels, y_pred), expected)
        self.assertAlmostEqual(scce(labels.reshape(-1, 1), y_pred), expected)
        np.testing.assert_allclose(scce.derivative(labels.reshape(-1, 1), y_pred),
                                   scce.derivative(labels, y_pred))

    def test_from_config_unknown_loss(self):
        with self.assertRaises(ValueError):
            LossFunction.from_config({'name': 'NotALoss'})


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
