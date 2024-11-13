import unittest

import numpy as np

from neuralnetlib.activations import Sigmoid
from neuralnetlib.layers import Layer, Dense, Activation


class TestLayers(unittest.TestCase):

    def test_layer_not_implemented(self):
        layer = Layer()
        with self.assertRaises(NotImplementedError):
            layer.forward_pass(np.array([1, 2, 3]))
        with self.assertRaises(NotImplementedError):
            layer.backward_pass(np.array([1, 2, 3]))

    def test_dense_layer(self):
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        output_size = 2
        dense = Dense(output_size)

        output = dense.forward_pass(input_data)
        self.assertEqual(output.shape, (input_data.shape[0], output_size))

        generator = np.random.default_rng(0)
        output_error = generator.random(output.shape)
        input_error = dense.backward_pass(output_error)
        self.assertEqual(input_error.shape, input_data.shape)

    def test_activation_layer(self):
        input_data = np.array([[-1, 0, 1]])
        activation_function = Sigmoid()
        activation = Activation(activation_function)

        output = activation.forward_pass(input_data)
        expected_output = activation_function(input_data)
        np.testing.assert_array_almost_equal(output, expected_output)

        generator = np.random.default_rng(0)
        output_error = generator.random(output.shape)
        input_error = activation.backward_pass(output_error)
        self.assertEqual(input_error.shape, input_data.shape)


if __name__ == '__main__':
    unittest.main()
