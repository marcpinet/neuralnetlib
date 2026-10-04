import os
import tempfile
import unittest

import numpy as np

from neuralnetlib.activations import Sigmoid
from neuralnetlib.callbacks import EarlyStopping
from neuralnetlib.layers import Input, Dense, Embedding, LSTM, GRU, Bidirectional, BatchNormalization, Conv2D, \
    Flatten, Reshape, Dropout
from neuralnetlib.models import Sequential, Activation, CategoricalCrossentropy, Autoencoder, Transformer, GAN
from neuralnetlib.optimizers import SGD, Adam


class TestSequential(unittest.TestCase):

    def setUp(self):
        self.model = Sequential()
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


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.directory = tempfile.mkdtemp()

    def test_all_parameters_are_trained(self):
        model = Sequential(random_state=0)
        model.add(Input(4))
        model.add(Embedding(10, 3))
        model.add(Bidirectional(LSTM(4, return_sequences=True)))
        model.add(GRU(3, return_sequences=True))
        model.add(LSTM(3))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        model.compile('bce', Adam(learning_rate=0.01))
        x = self.rng.integers(1, 10, (16, 4))
        y = self.rng.integers(0, 2, 16)
        model.forward_pass(x)

        before = [param.copy() for layer in model.layers for _, param, _ in layer.get_trainable_parameters()]
        model.fit(x, y, epochs=2, batch_size=8, verbose=False)
        after = [param for layer in model.layers for _, param, _ in layer.get_trainable_parameters()]

        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertFalse(np.allclose(old, new), "a parameter was not trained")

    def test_seeded_models_are_reproducible(self):
        def build_and_predict():
            model = Sequential(random_state=7)
            model.add(Input(5))
            model.add(Embedding(10, 4))
            model.add(Bidirectional(LSTM(3)))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile('bce', 'adam')
            x = np.arange(20).reshape(4, 5) % 10
            model.fit(x, np.array([0, 1, 0, 1]), epochs=2, batch_size=2, verbose=False)
            return model.predict(x)

        np.testing.assert_allclose(build_and_predict(), build_and_predict())

    def test_sparse_labels(self):
        model = Sequential(random_state=1)
        model.add(Input(4))
        model.add(Dense(3, activation='softmax'))
        model.compile('scce', Adam(learning_rate=0.05))
        x = self.rng.standard_normal((60, 4))
        y = np.argmax(x[:, :3], axis=1)
        history = model.fit(x, y, epochs=30, batch_size=20, verbose=False, metrics=['accuracy'])
        self.assertGreater(history['accuracy'][-1], 0.9)

    def test_save_and_load(self):
        model = Sequential(random_state=2)
        model.add(Input((6, 6, 1)))
        model.add(Conv2D(2, 3, padding='same'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Reshape((6, 12)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile('cce', 'adam')
        x = self.rng.random((8, 6, 6, 1))
        y = np.eye(2)[self.rng.integers(0, 2, 8)]
        model.fit(x, y, epochs=1, batch_size=4, verbose=False)

        filename = os.path.join(self.directory, 'model.json')
        model.save(filename)
        loaded = Sequential.load(filename)
        np.testing.assert_allclose(loaded.predict(x), model.predict(x))
        # the training can go on after saving (and after loading)
        model.fit(x, y, epochs=1, batch_size=4, verbose=False)
        loaded.fit(x, y, epochs=1, batch_size=4, verbose=False)

    def test_early_stopping_restores_first_epoch(self):
        model = Sequential(random_state=3)
        model.add(Input(2))
        model.add(Dense(1))
        model.compile('mse', SGD(learning_rate=1.0))  # diverges after the first epoch
        x = self.rng.standard_normal((32, 2)) * 10
        y = self.rng.standard_normal(32)
        weights = []

        class Recorder(EarlyStopping):
            def on_epoch_end(self, epoch, logs=None):
                weights.append(model.layers[1].weights.copy())
                return super().on_epoch_end(epoch, logs)

        model.fit(x, y, epochs=10, batch_size=32, verbose=False,
                  callbacks=[Recorder(patience=2, min_delta=0, restore_best_weights=True)])
        np.testing.assert_allclose(model.layers[1].weights, weights[0])

    def test_autoencoder(self):
        autoencoder = Autoencoder(random_state=0, skip_connections=True)
        autoencoder.add_encoder_layer(Input(8))
        autoencoder.add_encoder_layer(Dense(4, activation='relu'))
        autoencoder.add_encoder_layer(BatchNormalization())
        autoencoder.add_decoder_layer(Dense(8, activation='sigmoid'))
        autoencoder.compile(encoder_loss='mse', decoder_loss='mse', encoder_optimizer='adam', decoder_optimizer='adam')
        self.assertEqual(autoencoder.latent_dim, 4)
        x = self.rng.random((32, 8))
        history = autoencoder.fit(x, epochs=3, batch_size=8, verbose=False, validation_split=0.25)
        self.assertEqual(len(history['val_loss']), 3)
        self.assertEqual(autoencoder.predict(x).shape, (32, 8))

    def test_variational_autoencoder(self):
        autoencoder = Autoencoder(random_state=0, variational=True)
        autoencoder.add_encoder_layer(Input(8))
        autoencoder.add_encoder_layer(Dense(6, activation='linear'))
        autoencoder.add_decoder_layer(Dense(8, activation='sigmoid'))
        autoencoder.compile(encoder_loss='kld', decoder_loss='mse', encoder_optimizer='adam', decoder_optimizer='adam')
        x = self.rng.random((32, 8))
        autoencoder.fit(x, epochs=2, batch_size=8, verbose=False)
        self.assertEqual(autoencoder.latent_dim, 3)
        self.assertEqual(autoencoder.generate_image(x, n_samples=5, seed=0).shape, (5, 8))

    def test_transformer(self):
        model = Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=8, n_heads=2, n_encoder_layers=1,
                            n_decoder_layers=1, d_ff=16, max_sequence_length=6, random_state=0)
        model.compile(loss_function='cels', optimizer=Adam(learning_rate=0.01))
        x = [[2, 4, 5, 6, 3], [2, 7, 8, 3]]
        history = model.fit(x, x, epochs=5, batch_size=2, verbose=False)
        self.assertLess(history['loss'][-1], history['loss'][0])
        self.assertEqual(model.predict(np.array([[2, 4, 5, 6, 3, 0]]), max_length=6).shape[0], 1)

        filename = os.path.join(self.directory, 'transformer.json')
        model.save(filename)
        loaded = Transformer.load(filename)
        inputs = (np.array([[2, 4, 5, 6, 3, 0]]), np.array([[2, 4, 5, 6, 0, 0]]))
        np.testing.assert_allclose(loaded.forward_pass(inputs, training=False), model.forward_pass(inputs, training=False))

    def test_gan(self):
        generator = Sequential()
        generator.add(Input(4 + 2))
        generator.add(Dense(9, activation='sigmoid'))
        discriminator = Sequential()
        discriminator.add(Input(9 + 2))
        discriminator.add(Dense(1, activation='sigmoid'))
        gan = GAN(latent_dim=4, n_classes=2, random_state=0)
        gan.compile(generator, discriminator, generator_optimizer='adam', discriminator_optimizer='adam')

        first, _ = gan._generate_latent_points(4)
        second, _ = gan._generate_latent_points(4)
        self.assertFalse(np.allclose(first, second))

        x = self.rng.random((16, 9))
        y = np.eye(2)[self.rng.integers(0, 2, 16)]
        history = gan.fit(x, y, epochs=2, batch_size=8, n_critic=1, verbose=False)
        self.assertEqual(len(history['generator_loss']), 2)
        self.assertEqual(gan.predict(3, labels=np.array([0, 1, 1])).shape, (3, 9))

    def test_gan_is_reproducible(self):
        x = self.rng.random((16, 9))

        def train(metrics):
            generator = Sequential()
            generator.add(Input(4))
            generator.add(Dense(9, activation='sigmoid'))
            discriminator = Sequential()
            discriminator.add(Input(9))
            discriminator.add(Dense(1, activation='sigmoid'))
            gan = GAN(latent_dim=4, random_state=0)
            gan.compile(generator, discriminator, generator_optimizer='adam', discriminator_optimizer='adam')
            return gan.fit(x, epochs=2, batch_size=8, n_critic=1, metrics=metrics, verbose=False)['generator_loss']

        # the seed of the GAN is given to its sub-models, and computing metrics does not change the training
        np.testing.assert_allclose(train(None), train(None))
        np.testing.assert_allclose(train(['accuracy']), train(None))


if __name__ == '__main__':
    unittest.main()
