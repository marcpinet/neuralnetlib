import unittest

import numpy as np

from neuralnetlib.preprocessing import StandardScaler, PCA, TSNE, Tokenizer, CountVectorizer, ImageDataGenerator, \
    Imputer, pad_sequences, one_hot_encode
from neuralnetlib.learners import KMeans
from neuralnetlib.metrics import adjusted_rand_score
from neuralnetlib.utils import make_blobs, dict_with_ndarray_to_dict_with_list, dict_with_list_to_dict_with_ndarray


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_standard_scaler_constant_feature(self):
        X = np.hstack([self.rng.standard_normal((20, 2)), np.ones((20, 1))])
        self.assertFalse(np.isnan(StandardScaler().fit_transform(X)).any())

    def test_pca_images(self):
        X = self.rng.standard_normal((30, 3, 4))
        pca = PCA()
        projected = pca.fit_transform(X)
        self.assertEqual(projected.shape, (30, 12))
        np.testing.assert_allclose(pca.inverse_transform(projected), X)

    def test_tsne_separates_clusters(self):
        X, y = make_blobs(n_samples=60, centers=3, n_features=8, random_state=0, cluster_std=1.0)
        embedding = TSNE(perplexity=10, n_iter=300, random_state=0).fit_transform(X)
        labels = KMeans(n_clusters=3, random_state=0).fit_predict(embedding)
        self.assertGreater(adjusted_rand_score(labels, y), 0.9)

    def test_pad_sequences_with_arrays(self):
        padded = pad_sequences([np.array([5, 6, 7])], max_length=6)
        self.assertEqual(padded.tolist(), [[0, 0, 0, 5, 6, 7]])

    def test_one_hot_encode_float_indices(self):
        self.assertEqual(one_hot_encode(np.array([0.0, 2.0]), 3).tolist(), [[1, 0, 0], [0, 0, 1]])

    def test_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(["Hello, world!", "Hello there world"])
        sequences = tokenizer.texts_to_sequences(["Hello, world!"], preprocess_ponctuation=True)
        self.assertNotIn(tokenizer.UNK_IDX, sequences[0])
        self.assertEqual(tokenizer.word_docs['hello'], 2)
        text = tokenizer.sequences_to_texts([sequences[0] + [tokenizer.PAD_IDX] * 3])[0]
        self.assertNotIn(tokenizer.unk_token, text)
        self.assertNotIn(tokenizer.pad_token, text)

    def test_count_vectorizer_min_df(self):
        vectorizer = CountVectorizer(min_df=0.6).fit(["a cat", "a dog", "the cat sat"])
        self.assertNotIn('dog', vectorizer.vocabulary_)
        self.assertIn('cat', vectorizer.vocabulary_)

    def test_image_data_generator(self):
        image = np.zeros((16, 16))
        image[:, 8] = 1
        shifted = ImageDataGenerator(width_shift_range=4, random_state=1).random_transform(image)
        self.assertEqual(shifted.shape, image.shape)
        # a horizontal shift keeps the vertical line vertical
        self.assertEqual(np.count_nonzero(shifted.sum(axis=0)), 1)

        rotated = ImageDataGenerator(rotation_range=20, fill_mode='constant', random_state=0).random_transform(
            self.rng.random((16, 16, 1)))
        self.assertEqual(rotated.shape, (16, 16, 1))

        images = (self.rng.random((10, 8, 8)) * 255).astype(np.uint8)
        flow = ImageDataGenerator(rescale=1 / 255., random_state=0).flow(images, np.arange(10), batch_size=4, seed=0)
        seen = []
        for _ in range(3):
            batch_x, batch_y = next(flow)
            seen.extend(batch_y.tolist())
        self.assertGreater(batch_x.max(), 0)
        self.assertLessEqual(batch_x.max(), 1)
        self.assertEqual(sorted(seen), list(range(10)))

    def test_imputer_indicators(self):
        imputer = Imputer(strategy='mean', add_indicator=True).fit(np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]]))
        self.assertEqual(imputer.transform(np.array([[np.nan, 1.0]])).tolist(), [[1.5, 1.0, 1, 0]])

    def test_dict_conversions(self):
        state = {0: np.ones(2)}
        serialized = dict_with_ndarray_to_dict_with_list(state)
        self.assertIsInstance(state[0], np.ndarray)
        restored = dict_with_list_to_dict_with_ndarray({str(k): v for k, v in serialized.items()})
        self.assertIn(0, restored)


if __name__ == '__main__':
    unittest.main()
