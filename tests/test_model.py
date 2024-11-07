import unittest

import numpy as np

from neuralnetlib.activations import Sigmoid
from neuralnetlib.layers import Input, Dense
from neuralnetlib.model import Model, Activation, CategoricalCrossentropy
from neuralnetlib.optimizers import SGD


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.model.add(Input(10))
        self.model.add(Dense(20))
        self.model.add(Activation(Sigmoid()))
        self.model.compile(loss_function=CategoricalCrossentropy(), optimizer=SGD())

        rng = np.random.default_rng(0)
        self.x_train = rng.random((100, 10))
        self.y_train = rng.random((100, 20))
        self.x_test = rng.random((10, 10))
        self.y_test = rng.random((10, 20))

    def test_model_train_on_batch(self):
        loss = self.model.train_on_batch(self.x_train[:10], self.y_train[:10])
        self.assertIsInstance(loss, float)

    def test_model_train(self):
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=10, verbose=False)

    def test_model_evaluate(self):
        loss, preds = self.model.evaluate(self.x_test, self.y_test)
        self.assertIsInstance(loss, float)

    def test_model_predict(self):
        predictions = self.model.predict(self.x_test)
        self.assertEqual(predictions.shape, self.y_test.shape)


if __name__ == '__main__':
    unittest.main()
