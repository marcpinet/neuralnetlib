import json
import unittest

import numpy as np

from neuralnetlib.activations import Sigmoid, LeakyReLU, Softmax
from neuralnetlib.layers import Layer, Dense, Activation, Conv1D, Conv2D, Conv2DTranspose, MaxPooling1D, \
    MaxPooling2D, AveragePooling1D, AveragePooling2D, BatchNormalization, LayerNormalization, LSTM, GRU, \
    Bidirectional, Attention, MultiHeadAttention, UpSampling2D, Embedding, Permute, Reshape, Dropout, FeedForward, \
    TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        index = it.multi_index
        old_value = x[index]
        x[index] = old_value + eps
        f_plus = f()
        x[index] = old_value - eps
        f_minus = f()
        x[index] = old_value
        grad[index] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b)) / max(1e-8, np.max(np.abs(a)) + np.max(np.abs(b)))


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

    def test_activation_from_string(self):
        self.assertIsInstance(Activation('relu'), Activation)
        self.assertIsInstance(Activation('leaky_relu').activation_function, LeakyReLU)


class TestLayerGradients(unittest.TestCase):
    """The gradients computed by the backward passes are compared with finite differences."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def assert_gradients(self, layer, x, forward_kwargs=None, tol=1e-5):
        forward_kwargs = forward_kwargs or {}
        output = layer.forward_pass(x, **forward_kwargs)
        upstream = self.rng.standard_normal(output.shape)

        def loss():
            return np.sum(layer.forward_pass(x, **forward_kwargs) * upstream)

        layer.forward_pass(x, **forward_kwargs)
        input_gradient = layer.backward_pass(upstream)
        self.assertLess(relative_error(input_gradient, numerical_gradient(loss, x)), tol, "input gradient")

        for name, param, _ in layer.get_trainable_parameters():
            expected = numerical_gradient(loss, param)
            layer.forward_pass(x, **forward_kwargs)
            layer.backward_pass(upstream)
            gradient = {n: g for n, _, g in layer.get_trainable_parameters()}[name]
            if np.max(np.abs(expected)) < 1e-9:
                # e.g. the key bias of an attention layer has no effect on the output
                self.assertLess(np.max(np.abs(gradient)), 1e-7, name)
            else:
                self.assertLess(relative_error(np.reshape(gradient, param.shape), expected), tol, name)

    def test_dense(self):
        self.assert_gradients(Dense(4, random_state=1), self.rng.standard_normal((3, 5)))
        self.assert_gradients(Dense(4, random_state=1), self.rng.standard_normal((2, 3, 5)))

    def test_activations(self):
        for activation in ['sigmoid', 'relu', 'tanh', 'softmax', 'linear', 'leakyrelu', 'elu', 'selu', 'gelu']:
            self.assert_gradients(Activation(activation), self.rng.standard_normal((3, 5)) + 0.05)
        self.assert_gradients(Activation(Softmax()), self.rng.standard_normal((2, 3, 5)))

    def test_convolutions(self):
        for padding in ['valid', 'same']:
            for strides in [1, 2]:
                self.assert_gradients(Conv2D(3, 3, strides=strides, padding=padding, weights_init='he',
                                             bias_init='normal', random_state=2), self.rng.standard_normal((2, 7, 6, 2)))
                self.assert_gradients(Conv1D(3, 2, strides=strides, padding=padding, weights_init='he',
                                             bias_init='normal', random_state=2), self.rng.standard_normal((2, 7, 2)))
                self.assert_gradients(Conv2DTranspose(3, 3, strides=strides, padding=padding, weights_init='he',
                                                      bias_init='normal', random_state=2),
                                      self.rng.standard_normal((2, 4, 3, 2)))

    def test_pooling(self):
        for padding in ['valid', 'same']:
            for strides in [1, 2]:
                self.assert_gradients(MaxPooling2D(2, strides, padding), self.rng.standard_normal((2, 7, 6, 2)))
                self.assert_gradients(AveragePooling2D(3, strides, padding), self.rng.standard_normal((2, 7, 6, 2)))
                self.assert_gradients(MaxPooling1D(3, strides, padding), self.rng.standard_normal((2, 7, 3)))
                self.assert_gradients(AveragePooling1D(2, strides, padding), self.rng.standard_normal((2, 7, 3)))

    def test_same_padding_output_shape(self):
        x = self.rng.standard_normal((1, 28, 28, 1))
        self.assertEqual(Conv2D(2, 3, strides=2, padding='same').forward_pass(x).shape, (1, 14, 14, 2))
        self.assertEqual(MaxPooling2D(2, 2, padding='same').forward_pass(self.rng.standard_normal((1, 7, 7, 1))).shape,
                         (1, 4, 4, 1))
        self.assertEqual(Conv2DTranspose(2, 3, strides=2, padding='same').forward_pass(
            self.rng.standard_normal((1, 7, 7, 1))).shape, (1, 14, 14, 2))

    def test_normalizations(self):
        self.assert_gradients(BatchNormalization(), self.rng.standard_normal((6, 4)) * 3 + 1)
        self.assert_gradients(BatchNormalization(), self.rng.standard_normal((3, 4, 4, 2)))
        self.assert_gradients(LayerNormalization(), self.rng.standard_normal((2, 3, 4)))

    def test_batch_normalization_does_not_clip_inputs(self):
        bn = BatchNormalization()
        x = np.array([[0.0], [100.0], [200.0]])
        np.testing.assert_allclose(bn.forward_pass(x, training=True).ravel(), [-1.2247, 0, 1.2247], atol=1e-3)

    def test_recurrent_layers(self):
        self.assert_gradients(LSTM(3, return_sequences=True, random_state=3, clip_value=1e9),
                              self.rng.standard_normal((2, 4, 3)), tol=1e-4)
        self.assert_gradients(GRU(3, return_sequences=True, random_state=3, clip_value=1e9),
                              self.rng.standard_normal((2, 4, 3)) * 0.3, tol=1e-4)
        self.assert_gradients(Bidirectional(LSTM(3, return_sequences=False, random_state=3, clip_value=1e9)),
                              self.rng.standard_normal((2, 4, 3)), tol=1e-4)

    def test_attention_layers(self):
        self.assert_gradients(Attention(return_sequences=True), self.rng.standard_normal((2, 4, 3)))
        self.assert_gradients(Attention(return_sequences=False), self.rng.standard_normal((2, 4, 3)))
        self.assert_gradients(MultiHeadAttention(num_heads=2, key_dim=3, value_dim=4, random_state=1),
                              self.rng.standard_normal((2, 4, 6)))
        self.assert_gradients(MultiHeadAttention(num_heads=2, key_dim=3, normalize_attention=True, random_state=1),
                              self.rng.standard_normal((2, 4, 6)))
        mask = np.zeros((2, 1, 1, 4), dtype=bool)
        mask[0, ..., -1] = True
        self.assert_gradients(MultiHeadAttention(num_heads=2, key_dim=3, random_state=1),
                              self.rng.standard_normal((2, 4, 6)), {'mask': mask})

    def test_cross_attention(self):
        mha = MultiHeadAttention(num_heads=2, key_dim=3, random_state=1)
        query = self.rng.standard_normal((2, 3, 6))
        key_value = self.rng.standard_normal((2, 5, 6))
        output = mha.forward_pass((query, key_value, key_value))
        upstream = self.rng.standard_normal(output.shape)
        d_query, d_key, d_value = mha.backward_pass(upstream)

        def loss():
            return np.sum(mha.forward_pass((query, key_value, key_value)) * upstream)

        self.assertLess(relative_error(d_query, numerical_gradient(loss, query)), 1e-5)
        self.assertLess(relative_error(d_key + d_value, numerical_gradient(loss, key_value)), 1e-5)

    def test_transformer_layers(self):
        self.assert_gradients(FeedForward(8, 6, dropout_rate=0.0, random_state=1), self.rng.standard_normal((2, 3, 6)))
        self.assert_gradients(TransformerEncoderLayer(6, 2, 8, dropout_rate=0.0, attention_dropout=0.0, random_state=1),
                              self.rng.standard_normal((2, 3, 6)), tol=1e-4)

        decoder = TransformerDecoderLayer(6, 2, 8, dropout_rate=0.0, attention_dropout=0.0, random_state=1)
        x = self.rng.standard_normal((2, 3, 6))
        encoder_output = self.rng.standard_normal((2, 4, 6))
        output = decoder.forward_pass(x, encoder_output)
        upstream = self.rng.standard_normal(output.shape)
        dx, d_encoder_output = decoder.backward_pass(upstream)

        def loss():
            return np.sum(decoder.forward_pass(x, encoder_output) * upstream)

        self.assertLess(relative_error(dx, numerical_gradient(loss, x)), 1e-5)
        self.assertLess(relative_error(d_encoder_output, numerical_gradient(loss, encoder_output)), 1e-5)

    def test_upsampling(self):
        self.assert_gradients(UpSampling2D((2, 3)), self.rng.standard_normal((2, 3, 4, 2)))
        self.assert_gradients(UpSampling2D((2, 3), interpolation='bilinear'), self.rng.standard_normal((2, 3, 4, 2)))

    def test_embedding(self):
        embedding = Embedding(10, 4, random_state=0)
        indices = np.array([[1, 2, 0, 5], [5, 5, 3, 0]])
        output = embedding.forward_pass(indices.astype(float))
        upstream = self.rng.standard_normal(output.shape)
        embedding.backward_pass(upstream)
        expected = numerical_gradient(lambda: np.sum(embedding.forward_pass(indices) * upstream), embedding.weights)
        expected[0] = 0  # the padding index is not trained
        np.testing.assert_allclose(embedding.d_weights, expected, atol=1e-6)


class TestLayerBehaviour(unittest.TestCase):

    def test_dropout_masks_change(self):
        dropout = Dropout(0.5, random_state=42)
        x = np.ones((10, 10))
        self.assertFalse(np.array_equal(dropout.forward_pass(x), dropout.forward_pass(x)))
        np.testing.assert_array_equal(dropout.forward_pass(x, training=False), x)
        np.testing.assert_array_equal(dropout.backward_pass(x), x)

    def test_sibling_dropouts_use_different_masks(self):
        encoder = TransformerEncoderLayer(8, 2, 16, dropout_rate=0.5, random_state=0)
        x = np.ones((2, 3, 8))
        self.assertFalse(np.array_equal(encoder.attention_dropout.forward_pass(x), encoder.ffn_dropout.forward_pass(x)))

    def test_positional_encoding_odd_dimension(self):
        pe = PositionalEncoding(10, 5)
        self.assertEqual(pe.forward_pass(np.zeros((2, 4, 5))).shape, (2, 4, 5))

    def test_permute(self):
        x = np.arange(24).reshape(2, 3, 4)
        permute = Permute((2, 1))
        self.assertEqual(permute.forward_pass(x).shape, (2, 4, 3))
        self.assertEqual(Permute.from_config(json.loads(json.dumps(permute.get_config()))).dims, (2, 1))

    def test_config_round_trips(self):
        x = np.random.default_rng(0).standard_normal((2, 3, 8))

        for layer in [FeedForward(16, 8, random_state=0), TransformerEncoderLayer(8, 2, 16, random_state=0),
                      Reshape((4, 6)), Bidirectional(LSTM(4, return_sequences=True, random_state=0)),
                      MultiHeadAttention(2, 4, use_bias=False, random_state=0)]:
            expected = layer.forward_pass(x, training=False) if not isinstance(layer, (Reshape,)) else layer.forward_pass(x)
            config = json.loads(json.dumps(layer.get_config()))
            loaded = Layer.from_config(config)
            output = loaded.forward_pass(x, training=False) if not isinstance(layer, (Reshape,)) else loaded.forward_pass(x)
            np.testing.assert_allclose(output, expected, err_msg=type(layer).__name__)

    def test_activation_config_keeps_parameters(self):
        layer = Activation(LeakyReLU(alpha=0.3))
        loaded = Activation.from_config(json.loads(json.dumps(layer.get_config())))
        self.assertEqual(loaded.activation_function.alpha, 0.3)


if __name__ == '__main__':
    unittest.main()
