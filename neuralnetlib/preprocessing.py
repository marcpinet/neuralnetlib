import random
import re
import numpy as np

from time import time_ns
from enum import Enum
from collections import defaultdict
from collections.abc import Generator


def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode 1D or 2D indices. One hot encoded labels are binary vectors representing categorical values,
    with exactly one high (or "hot" = 1) bit indicating the presence of a specific category
    and all other bits low (or "cold" = 0)."""
    if indices.ndim == 1:
        one_hot = np.zeros((indices.size, num_classes))
        one_hot[np.arange(indices.size), indices] = 1
    elif indices.ndim == 2:
        batch_size, seq_len = indices.shape
        one_hot = np.zeros((batch_size, seq_len, num_classes))
        rows, cols = np.meshgrid(
            np.arange(batch_size), np.arange(seq_len), indexing='ij')
        one_hot[rows, cols, indices] = 1
    else:
        raise ValueError("Unsupported input shape. Expected 1D or 2D indices.")
    return one_hot


def apply_threshold(y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Applies a threshold to the predictions. Typically used for binary classification."""
    return (y_pred > threshold).astype(int)


def im2col_2d(input_data: np.ndarray, filter_h: int, filter_w: int, stride: int | tuple[int, int] = 1,
              pad: int | tuple[int, int | float] = 0) -> np.ndarray:
    """Transform 4 dimensional images to 2 dimensional array.

    Args:
        input_data (np.ndarray): 4D input images (batch_size, height, width, channels)
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int | tuple[int, int], optional): the interval of stride. Defaults to 1.
        pad (int | tuple[int, int | float], optional): the interval of padding. Defaults to 0.

    Returns:
        np.ndarray: A 2D array of shape (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, H, W, C = input_data.shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = map(int, pad)

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    pad_width = [(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)]
    padded_input = np.pad(input_data, pad_width, mode='constant')

    col = np.zeros((N, out_h, out_w, filter_h, filter_w, C))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            col[:, :, :, y, x, :] = padded_input[:,
                                                 y:y_max:stride_h,
                                                 x:x_max:stride_w,
                                                 :]

    col = col.transpose(0, 1, 2, 5, 3, 4).reshape(N * out_h * out_w, -1)

    return col


def col2im_2d(col: np.ndarray, input_shape: tuple[int, int, int, int], filter_h: int, filter_w: int,
              stride: int | tuple[int, int] = 1, pad: int | tuple[int, int | float] = 0) -> np.ndarray:
    """
    Inverse of im2col.

    Args:
        col (np.ndarray): 2D array of shape (N*out_h*out_w, C*filter_h*filter_w)
        input_shape (tuple): the shape of original input images (N, H, W, C)
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int or tuple): the interval of stride
        pad (int or tuple): the interval of padding

    Returns:
        image (np.ndarray): original images in NHWC format
    """
    N, H, W, C = input_shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = map(int, pad)

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)

    img_h = H + 2 * pad_h + stride_h - 1
    img_w = W + 2 * pad_w + stride_w - 1

    img = np.zeros((N, img_h, img_w, C))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            img[:, y:y_max:stride_h, x:x_max:stride_w, :] += col[:, :, :, :, y, x]

    return img[:, pad_h:pad_h + H, pad_w:pad_w + W, :]


def im2col_1d(input_data: np.ndarray, filter_size: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Transform 3 dimensional images to 2 dimensional array in NLC format.

    Args:
        input_data (np.ndarray): 3 dimensional input images (N, L, C)
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        col (np.ndarray): 2 dimensional array
    """
    N, L, C = input_data.shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    padded_input = np.pad(
        input_data, ((0, 0), (pad, pad), (0, 0)), mode='constant')

    col = np.zeros((N, out_l, filter_size, C))

    for y in range(filter_size):
        y_max = y + stride * out_l
        col[:, :, y, :] = padded_input[:, y:y_max:stride, :]

    col = col.reshape(N * out_l, -1)
    return col


def col2im_1d(col: np.ndarray, input_shape: tuple[int, int, int], filter_size: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Inverse of im2col_1d for NLC format.

    Args:
        col (np.ndarray): 2 dimensional array
        input_shape (tuple): the shape of original input images (N, L, C)
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        image (np.ndarray): original images in NLC format
    """
    N, L, C = input_shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    col = col.reshape(N, out_l, filter_size, C)

    image = np.zeros((N, L + 2 * pad + stride - 1, C))

    for y in range(filter_size):
        y_max = y + stride * out_l
        image[:, y:y_max:stride, :] += col[:, :, y, :]

    return image[:, pad:L + pad, :]


def pad_sequences(sequences: np.ndarray, max_length: int, pad_value: int = 0, padding: str = 'pre', truncating: str = 'pre') -> np.ndarray:
    """
    Pads or truncates sequences to a specified maximum length.

    Args:
        sequences (list of list or np.ndarray): Input sequences of varying lengths.
        max_length (int): Maximum length of the output sequences.
        pad_value (int/float): Value to use for padding.
        padding (str): 'pre' or 'post', to pad before or after the sequence.
        truncating (str): 'pre' or 'post', to truncate the sequence at the beginning or end.

    Returns:
        np.ndarray: Padded or truncated sequences as a NumPy array.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            if truncating == 'pre':
                seq = seq[-max_length:]
            else:
                seq = seq[:max_length]

        if padding == 'pre':
            padded_seq = [pad_value] * (max_length - len(seq)) + seq
        else:
            padded_seq = seq + [pad_value] * (max_length - len(seq))

        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)


def clip_gradients(gradients: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(gradients)
    if norm > threshold:
        gradients = threshold * gradients / norm
    return gradients


def normalize_gradient(gradients: np.ndarray, scale: float = 1.0) -> np.ndarray:
    axis = tuple(range(1, gradients.ndim))
    grad_norm = np.sqrt(np.sum(gradients ** 2, axis=axis, keepdims=True) + 1e-8)
    
    grad_norm = np.maximum(grad_norm, 1e-8)
    
    normalized_gradients = gradients / grad_norm
    return normalized_gradients * scale


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.EPSILON = 1e-8

    def fit(self, X: np.ndarray) -> None:
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_
        self.scale_ = np.where(self.scale_ == 0, self.EPSILON, self.scale_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * self.scale_ + self.min_


class PCA:
    def __init__(self, n_components: int = None, random_state: int = None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None
        self.mean = None
        self.input_shape = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray):
        if self.n_components is None:
            self.n_components = X.shape[1]

        self.input_shape = X.shape[1:]
        X = X.reshape(X.shape[0], -1)

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

        explained_variance = np.var(
            np.dot(X_centered, self.components), axis=0)
        total_variance = np.sum(np.var(X_centered, axis=0))

        self.explained_variance_ratio = explained_variance / total_variance

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        X_centered = X - self.mean

        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X_reconstructed = np.dot(X, self.components.T) + self.mean
        return X_reconstructed.reshape((-1, *self.input_shape))


class TSNE:
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, learning_rate: float = 200.0, n_iter: int = 1000, random_state: int = None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding_ = None
        self.kl_div = None

    def _calculate_pairwise_affinities(self, X):
        distances = np.sum(
            (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        P = np.exp(-distances / (2 * self.perplexity ** 2))
        np.fill_diagonal(P, 0)
        P /= np.sum(P, axis=1, keepdims=True)
        return P

    def _kl_divergence(self, P, Q):
        return np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))

    def fit_transform(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        P = self._calculate_pairwise_affinities(X)
        rng = np.random.default_rng(self.random_state)
        Y = rng.standard_normal((n_samples, self.n_components)) * 1e-4

        for i in range(self.n_iter):
            distances = np.sum(
                (Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
            Q = 1 / (1 + distances)
            np.fill_diagonal(Q, 0)
            Q /= np.sum(Q)

            PQ_diff = (P - Q) * Q
            grad = np.zeros_like(Y)
            for j in range(n_samples):
                grad[j] = np.sum(
                    (Y[j] - Y) * PQ_diff[j, :, np.newaxis], axis=0)

            Y -= self.learning_rate * grad

            if (i + 1) % 100 == 0:
                kl_div = self._kl_divergence(P, Q)
                self.kl_div = kl_div

        self.embedding_ = Y
        return self.embedding_


class Tokenizer:
    def __init__(self,
                 num_words: int | None = None,
                 filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower: bool = True,
                 split: str = ' ',
                 mode: str = 'word',
                 bpe_merges: int = None,
                 pad_token: str = "<PAD>",
                 unk_token: str = "<UNK>",
                 sos_token: str = "<SOS>",
                 eos_token: str = "<EOS>") -> None:
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.mode = mode
        self.bpe_merges = bpe_merges

        if mode not in ['char', 'word', 'bpe']:
            raise ValueError("Mode must be one of 'char', 'word', or 'bpe'")

        if mode == 'bpe' and bpe_merges is None:
            raise ValueError(
                "bpe_merges must be specified when using BPE mode")

        self.SPECIAL_TOKENS = {
            'PAD': (pad_token, 0),
            'UNK': (unk_token, 1),
            'SOS': (sos_token, 2),
            'EOS': (eos_token, 3)
        }

        for token_name, (token_text, token_idx) in self.SPECIAL_TOKENS.items():
            setattr(self, f"{token_name}_IDX", token_idx)
            setattr(self, f"{token_name.lower()}_token", token_text)

        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}
        self.word_docs = {}
        self.document_count = 0
        self.bpe_cache = {}

        for token, (text, idx) in self.SPECIAL_TOKENS.items():
            self.word_index[text] = idx
            self.index_word[idx] = text

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r"([!\"#$%&()*+,-./:;<=>?@\[\]^_`{|}~])", r" \1 ", text)
        text = re.sub(r"(\b\w)'(\w)", r"\1' \2", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_pairs(self, word: list[str]) -> set[tuple[str, str]]:
        """Get all adjacent pairs in the word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def learn_bpe(self, texts: list[str], cache: bool = True) -> dict:
        vocab = defaultdict(int)
        pairs = defaultdict(int)

        for text in texts:
            for word in text.split():
                if self.lower:
                    word = word.lower()
                chars = list(word)
                vocab[tuple(chars)] += 1

                word_pairs = self.get_pairs(chars)
                for pair in word_pairs:
                    pairs[pair] += 1

        merges = {}
        for i in range(self.bpe_merges):
            if not pairs:
                break
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            merges[best_pair] = i

            new_vocab = {}
            for word, freq in vocab.items():
                word = list(word)
                while True:
                    changes = 0
                    for j in range(len(word)-1):
                        if tuple(word[j:j+2]) == best_pair:
                            word[j:j+2] = [''.join(best_pair)]
                            changes += 1
                    if changes == 0:
                        break
                new_vocab[tuple(word)] = freq
            vocab = new_vocab

            pairs = defaultdict(int)
            for word, freq in vocab.items():
                word_pairs = self.get_pairs(list(word))
                for pair in word_pairs:
                    pairs[pair] += freq

        self.bpe_merges = merges
        return vocab

    def bpe_encode(self, text: str) -> list[str]:
        if not self.bpe_merges:
            return list(text)

        if self.lower:
            text = text.lower()

        if text in self.bpe_cache:
            return self.bpe_cache[text]

        word = list(text)
        pairs = self.get_pairs(word)

        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_merges.get(
                pair, float('inf')))
            if bigram not in self.bpe_merges:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and tuple(word[i:i+2]) == bigram:
                    new_word.append(''.join(bigram))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self.get_pairs(word)

        self.bpe_cache[text] = word
        return word

    def fit_on_texts(self, texts: list[str], preprocess_ponctuation: bool = True) -> None:
        FIRST_REGULAR_IDX = len(self.SPECIAL_TOKENS)

        processed_texts = [self.preprocess_text(text) if preprocess_ponctuation else text
                           for text in texts]

        if self.mode == 'bpe':
            self.learn_bpe(processed_texts)

        for text in processed_texts:
            self.document_count += 1

            if self.mode == 'char':
                seq = list(text)
            elif self.mode == 'bpe':
                seq = []
                for word in text.split():
                    seq.extend(self.bpe_encode(word))
            else:
                seq = text.split(self.split)

            for w in seq:
                if self.lower:
                    w = w.lower()
                if w in self.filters:
                    continue
                if w in [token for token, _ in self.SPECIAL_TOKENS.values()]:
                    continue

                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                self.word_docs[w] = self.word_docs.get(w, 0) + 1

        wcounts = sorted(self.word_counts.items(),
                         key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]

        next_index = FIRST_REGULAR_IDX
        for w in sorted_voc:
            if w not in self.word_index:
                self.word_index[w] = next_index
                next_index += 1

        if self.num_words is not None:
            keep_tokens = {w: i for w, i in self.word_index.items()
                           if (i < self.num_words or
                               w in [token for token, _ in self.SPECIAL_TOKENS.values()])}
            self.word_index = keep_tokens

        self.index_word = {i: w for w, i in self.word_index.items()}

    def texts_to_sequences(self, texts: list[str],
                           preprocess_ponctuation: bool = False,
                           add_special_tokens: bool = True) -> list[list[int]]:
        sequences = []
        for text in texts:
            if preprocess_ponctuation:
                text = self.preprocess_text(text)

            if self.mode == 'char':
                seq = list(text)
            elif self.mode == 'bpe':
                seq = []
                for word in text.split():
                    seq.extend(self.bpe_encode(word))
            else:
                seq = text.split(self.split)

            vect = []
            for w in seq:
                if self.lower:
                    w = w.lower()

                i = self.word_index.get(w)

                if i is not None:
                    if self.num_words and i >= self.num_words:
                        if w in {self.pad_token, self.unk_token, self.sos_token, self.eos_token}:
                            vect.append(i)
                        else:
                            vect.append(self.UNK_IDX)
                    else:
                        vect.append(i)
                else:
                    if '-' in w and self.mode == 'word':
                        subwords = w.split('-')
                        for idx, subw in enumerate(subwords):
                            if idx > 0:
                                vect.append(self.word_index.get(
                                    '-', self.UNK_IDX))
                            i = self.word_index.get(subw, self.UNK_IDX)
                            vect.append(i)
                    else:
                        vect.append(self.UNK_IDX)

            if add_special_tokens:
                vect = [self.SOS_IDX] + vect + [self.EOS_IDX]
            sequences.append(vect)

        return sequences

    def sequences_to_texts(self, sequences: list[list[int]]) -> list[str]:
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences: list[list[int]]) -> Generator[str, None, None]:
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None and word not in {self.pad_token}:
                    vect.append(word)
                else:
                    vect.append(self.unk_token)

            if self.mode == 'char':
                yield ''.join(vect)
            else:
                yield ' '.join(vect)

    def get_vocab_size(self) -> int:
        if self.num_words is not None:
            return min(len(self.word_index), self.num_words)
        return len(self.word_index)

    def get_config(self) -> dict:
        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'mode': self.mode,
            'bpe_merges': self.bpe_merges if hasattr(self, 'bpe_merges') else None,
            'document_count': self.document_count,
        }


class CountVectorizer:
    def __init__(self, lowercase: bool = True, token_pattern: str = r'(?u)\b\w\w+\b', max_df: float | int = 1.0,
                 min_df: float | int = 1, max_features: int | None = None) -> None:
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self.document_count_ = 0

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(self.token_pattern, text)

    def fit(self, raw_documents: list[str]) -> "CountVectorizer":
        self.document_count_ = len(raw_documents)
        term_freq = {}
        doc_freq = {}

        for doc in raw_documents:
            term_counts = {}
            for term in self._tokenize(doc):
                if term not in term_counts:
                    term_counts[term] = 1
                else:
                    term_counts[term] += 1

            for term, count in term_counts.items():
                if term not in term_freq:
                    term_freq[term] = count
                    doc_freq[term] = 1
                else:
                    term_freq[term] += count
                    doc_freq[term] += 1

        if isinstance(self.max_df, float):
            max_doc_count = int(self.max_df * self.document_count_)
        else:
            max_doc_count = self.max_df

        if isinstance(self.min_df, float):
            min_doc_count = int(self.min_df * self.document_count_)
        else:
            min_doc_count = self.min_df

        terms = [term for term, freq in doc_freq.items()
                 if min_doc_count <= freq <= max_doc_count]

        if self.max_features is not None:
            terms = sorted(terms, key=lambda t: term_freq[t], reverse=True)[
                :self.max_features]

        self.vocabulary_ = {term: idx for idx,
                            term in enumerate(sorted(terms))}

        return self

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        X = np.zeros((len(raw_documents), len(self.vocabulary_)), dtype=int)

        for doc_idx, doc in enumerate(raw_documents):
            for term in self._tokenize(doc):
                if term in self.vocabulary_:
                    X[doc_idx, self.vocabulary_[term]] += 1

        return X

    def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get))

    def get_vocabulary(self) -> dict:
        return dict(sorted(self.vocabulary_.items(), key=lambda x: x[1]))


class NGram:
    def __init__(self,
                 n: int = 3,
                 token_type: str = "char",
                 start_token: str = '$',
                 end_token: str = '^',
                 separator: str = ' '):

        self.n = n
        self.token_type = token_type
        self.start_token = start_token
        self.end_token = end_token
        self.separator = separator
        self.ngrams = defaultdict(list)
        self.transitions = defaultdict(list)

    def _tokenize(self, text: str) -> list[str]:
        if self.token_type == "char":
            return list(text)
        return text.split(self.separator)

    def _join_tokens(self, tokens: list[str]) -> str:
        if self.token_type == "char":
            return ''.join(tokens)
        return self.separator.join(tokens)

    def _process_sequence(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return ([self.start_token] * (self.n - 1)) + tokens + [self.end_token]

    def fit(self, sequences: list[str]) -> "NGram":
        self.ngrams.clear()
        self.transitions.clear()

        for sequence in sequences:
            processed_seq = self._process_sequence(sequence)

            for i in range(len(processed_seq) - self.n + 1):
                context = tuple(processed_seq[i:i + self.n - 1])
                target = processed_seq[i + self.n - 1]
                self.ngrams[context].append(target)

            tokens = self._tokenize(sequence)
            for i in range(len(tokens) - 1):
                current_token = tokens[i]
                next_token = tokens[i + 1]

                if (current_token != self.start_token and
                        current_token != self.end_token and
                        next_token != self.start_token and
                        next_token != self.end_token):
                    self.transitions[current_token].append(next_token)

        return self

    def _get_random_start(self) -> list[str]:
        if self.token_type == "char":
            return [self.start_token] * (self.n - 1)

        start_contexts = [
            context for context in self.ngrams.keys()
            if (context[0] == self.start_token and
                self.end_token not in context)
        ]

        if not start_contexts:
            return [self.start_token] * (self.n - 1)

        chosen_context = random.choice(start_contexts)
        return list(chosen_context)

    def generate_sequence(self, min_length: int = 5, max_length: int = None, variability: float = 0.3) -> str:
        if not self.ngrams:
            raise ValueError("Model not trained. Call fit() first.")

        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            current = self._get_random_start()

            while True:
                context = tuple(current[-(self.n - 1):])

                if context not in self.ngrams:
                    if (self.token_type == "word" and
                            current[-1] in self.transitions):
                        next_token = random.choice(
                            self.transitions[current[-1]])
                        current.append(next_token)
                        continue
                    break

                next_token = random.choice(self.ngrams[context])
                current.append(next_token)

                if next_token == self.end_token:
                    sequence = current[(self.n - 1):-1]
                    if len(sequence) >= min_length:
                        if max_length is None or len(sequence) <= max_length:
                            result = self._join_tokens(sequence)
                            if self.token_type == "word":
                                result = result.capitalize()
                            return result
                    break

                if max_length and len(current) - (self.n - 1) > max_length:
                    break

                if (self.token_type == "word" and
                        random.random() < variability and
                        current[-1] in self.transitions):
                    next_token = random.choice(self.transitions[current[-1]])
                    current.append(next_token)

        raise ValueError(
            f"Could not generate a sequence after {max_attempts} attempts.")

    def generate_sequences(self,
                           n_sequences: int = 20,
                           min_length: int = 5,
                           max_length: int = None) -> list[str]:
        sequences = []
        attempts = 0
        max_attempts = n_sequences * 2

        while len(sequences) < n_sequences and attempts < max_attempts:
            attempts += 1
            try:
                sequence = self.generate_sequence(min_length, max_length)
                if sequence not in sequences:
                    sequences.append(sequence)
            except ValueError:
                continue

        return sequences

    def get_contexts(self) -> dict:
        return dict(self.ngrams)


class ImageDataGenerator:
    def __init__(
        self,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        horizontal_flip=False,
        vertical_flip=False,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        rescale=None,
        random_state=None
    ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.rescale = rescale
        self.random_state = random_state if random_state is not None else time_ns()
        self.rng = np.random.default_rng(self.random_state)

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        else:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

    def random_transform(self, x, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)

        img_row_axis, img_col_axis, img_channel_axis = 0, 1, 2
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]

        transform_matrix = np.eye(3)

        if self.rotation_range:
            theta = rng.uniform(-self.rotation_range, self.rotation_range)
            rotation_matrix = self._get_rotation_matrix(theta)
            transform_matrix = np.dot(transform_matrix, rotation_matrix)

        if self.width_shift_range or self.height_shift_range:
            tx = 0
            ty = 0
            if self.width_shift_range:
                if isinstance(self.width_shift_range, int):
                    tx = rng.integers(-self.width_shift_range,
                                      self.width_shift_range + 1)
                else:
                    tx = rng.uniform(-self.width_shift_range,
                                     self.width_shift_range) * w
            if self.height_shift_range:
                if isinstance(self.height_shift_range, int):
                    ty = rng.integers(-self.height_shift_range,
                                      self.height_shift_range + 1)
                else:
                    ty = rng.uniform(-self.height_shift_range,
                                     self.height_shift_range) * h

            translation_matrix = np.array([[1, 0, tx],
                                           [0, 1, ty],
                                           [0, 0, 1]])
            transform_matrix = np.dot(transform_matrix, translation_matrix)

        if self.zoom_range[0] != 1 or self.zoom_range[1] != 1:
            zx = rng.uniform(self.zoom_range[0], self.zoom_range[1])
            zy = zx
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if not np.array_equal(transform_matrix, np.eye(3)):
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transforms = []
            for i in range(x.shape[img_channel_axis]):
                transforms.append(self._affine_transform(
                    x[..., i],
                    transform_matrix,
                    fill_mode=self.fill_mode,
                    cval=self.cval))
            x = np.stack(transforms, axis=-1)

        if self.horizontal_flip and rng.random() < 0.5:
            x = x[:, ::-1]
        if self.vertical_flip and rng.random() < 0.5:
            x = x[::-1]

        if self.brightness_range is not None:
            brightness = rng.uniform(self.brightness_range[0],
                                     self.brightness_range[1])
            x = x * brightness

        if self.channel_shift_range != 0:
            x = self._channel_shift(x, self.channel_shift_range, rng)

        if self.rescale is not None:
            x *= self.rescale

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        n = x.shape[0]
        batch_index = 0
        index_array = np.arange(n)

        while True:
            if shuffle:
                rng.shuffle(index_array)

            current_index = (batch_index * batch_size) % n

            if n > current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = n - current_index

            batch_index += 1
            batch_indices = index_array[current_index:
                                        current_index + current_batch_size]

            batch_x = np.zeros((current_batch_size,) + x.shape[1:],
                               dtype=x.dtype)

            for i, j in enumerate(batch_indices):
                x_aug = self.random_transform(x[j])
                batch_x[i] = x_aug

            if y is None:
                yield batch_x
            else:
                batch_y = y[batch_indices]
                yield batch_x, batch_y

    def _get_rotation_matrix(self, theta):
        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])

    def _affine_transform(self, x, matrix, fill_mode='nearest', cval=0.0):
        h, w = x.shape[:2]

        y_coords, x_coords = np.meshgrid(
            np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([y_coords, x_coords, np.ones_like(x_coords)])
        coords_reshaped = coords.reshape(3, -1)

        matrix_inv = np.linalg.inv(matrix)
        transformed_coords = np.dot(matrix_inv, coords_reshaped)

        y_coords = transformed_coords[0].reshape(h, w)
        x_coords = transformed_coords[1].reshape(h, w)

        if fill_mode == 'nearest':
            y_coords = np.clip(np.round(y_coords), 0, h - 1).astype(np.int32)
            x_coords = np.clip(np.round(x_coords), 0, w - 1).astype(np.int32)
            return x[y_coords, x_coords]

        elif fill_mode == 'constant':
            y_floor = np.floor(y_coords).astype(np.int32)
            y_ceil = y_floor + 1
            x_floor = np.floor(x_coords).astype(np.int32)
            x_ceil = x_floor + 1

            valid_coords = (y_floor >= 0) & (
                y_ceil < h) & (x_floor >= 0) & (x_ceil < w)

            y_floor = np.clip(y_floor, 0, h-1)
            y_ceil = np.clip(y_ceil, 0, h-1)
            x_floor = np.clip(x_floor, 0, w-1)
            x_ceil = np.clip(x_ceil, 0, w-1)

            dy = y_coords - y_floor
            dx = x_coords - x_floor

            dy = dy[..., np.newaxis]
            dx = dx[..., np.newaxis]

            values = (
                x[y_floor, x_floor] * (1 - dy) * (1 - dx) +
                x[y_ceil, x_floor] * dy * (1 - dx) +
                x[y_floor, x_ceil] * (1 - dy) * dx +
                x[y_ceil, x_ceil] * dy * dx
            )

            return np.where(valid_coords[..., np.newaxis], values, cval)

        elif fill_mode == 'reflect':
            y_coords = np.clip(y_coords, -h, 2*h-1)
            x_coords = np.clip(x_coords, -w, 2*w-1)
            y_coords = np.where(y_coords < 0, -y_coords, y_coords)
            x_coords = np.where(x_coords < 0, -x_coords, x_coords)
            y_coords = np.where(y_coords >= h, 2*h - y_coords - 2, y_coords)
            x_coords = np.where(x_coords >= w, 2*w - x_coords - 2, x_coords)
            y_coords = y_coords.astype(np.int32)
            x_coords = x_coords.astype(np.int32)
            return x[y_coords, x_coords]

        elif fill_mode == 'wrap':
            y_coords = np.remainder(y_coords, h).astype(np.int32)
            x_coords = np.remainder(x_coords, w).astype(np.int32)
            return x[y_coords, x_coords]

        return x

    def _channel_shift(self, x, intensity, rng):
        x = np.array(x, copy=True)
        channels = x.shape[-1] if x.ndim > 2 else 1
        for i in range(channels):
            shift = rng.uniform(-intensity, intensity)
            if x.ndim > 2:
                x[..., i] = np.clip(x[..., i] + shift, 0, 1)
            else:
                x = np.clip(x + shift, 0, 1)
        return x


class SpectralNorm:
    def __init__(self, n_power_iterations: int = 1, random_state: int = None):
        self.n_power_iterations = n_power_iterations
        self.u_dict = {}
        self.v_dict = {}
        self.rng = np.random.default_rng(
            random_state if random_state is not None else time_ns())

    def _get_uv_key(self, W: np.ndarray) -> tuple:
        return tuple(W.shape)

    def _initialize_uv(self, W: np.ndarray, key: tuple):
        if len(W.shape) == 1:
            height = W.shape[0]
            width = 1
            W = W.reshape(-1, 1)
        else:
            height, width = W.shape

        self.u_dict[key] = self.rng.normal(0, 1, (height, 1))
        self.u_dict[key] = self.u_dict[key] / np.linalg.norm(self.u_dict[key])

        self.v_dict[key] = self.rng.normal(0, 1, (width, 1))
        self.v_dict[key] = self.v_dict[key] / np.linalg.norm(self.v_dict[key])

    def __call__(self, W: np.ndarray) -> np.ndarray:
        if W is None:
            return None

        original_shape = W.shape

        if len(original_shape) == 1:
            W = W.reshape(-1, 1)

        if W.size < 2:
            return W.reshape(original_shape)

        key = self._get_uv_key(W)

        if key not in self.u_dict:
            self._initialize_uv(W, key)

        u = self.u_dict[key]
        v = self.v_dict[key]

        for _ in range(self.n_power_iterations):
            v = W.T @ u
            v = v / (np.linalg.norm(v) + 1e-12)
            u = W @ v
            u = u / (np.linalg.norm(u) + 1e-12)

        self.u_dict[key] = u
        self.v_dict[key] = v

        sigma = float(u.T @ W @ v)
        normalized_W = W / (sigma + 1e-12)

        return normalized_W.reshape(original_shape)

    def reset(self):
        """Reset stored u,v vectors"""
        self.u_dict.clear()
        self.v_dict.clear()


class Strategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    RANDOM = "random"


class Imputer:
    def __init__(self, strategy: str = "mean", fill_value: float = None, add_indicator: bool = False, random_state: int = None):
        if isinstance(strategy, str):
            strategy = Strategy(strategy)

        self.strategy: Strategy = strategy
        self.fill_value: float = fill_value
        self.add_indicator: bool = add_indicator
        self.random_state: int = random_state
        self.statistics_: dict[int, float] = {}
        self.indicators_: dict[int, np.ndarray] = {}
        self.is_1d_: bool = False

        if strategy == Strategy.RANDOM and random_state is not None:
            np.random.seed(random_state)

    def _compute_mode(self, column: np.ndarray) -> float:
        unique_vals, counts = np.unique(
            column[~np.isnan(column)], return_counts=True)
        return unique_vals[np.argmax(counts)]

    def _compute_statistics(self, X: np.ndarray, column_idx: int) -> float:
        non_missing = X[~np.isnan(X[:, column_idx]), column_idx]

        if len(non_missing) == 0:
            raise ValueError(f"Column {column_idx} has no non-missing values")

        if self.strategy == Strategy.MEAN:
            return float(np.mean(non_missing))
        elif self.strategy == Strategy.MEDIAN:
            return float(np.median(non_missing))
        elif self.strategy == Strategy.MODE:
            return float(self._compute_mode(non_missing))
        elif self.strategy == Strategy.CONSTANT:
            return self.fill_value
        elif self.strategy == Strategy.RANDOM:
            self.random_params_[column_idx] = non_missing
            return None

    def fit(self, X: np.ndarray) -> "Imputer":
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.is_1d_ = X.ndim == 1
        if self.is_1d_:
            X = X.reshape(-1, 1)

        self.random_params_: dict[int, np.ndarray] = {}

        if self.add_indicator:
            self.indicators_ = {
                i: np.isnan(X[:, i]) for i in range(X.shape[1])
            }

        for i in range(X.shape[1]):
            self.statistics_[i] = self._compute_statistics(X, i)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(-1, 1)

        X_imputed = X.copy()

        if self.strategy in [Strategy.MEAN, Strategy.MEDIAN, Strategy.MODE, Strategy.CONSTANT]:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                X_imputed[mask, i] = self.statistics_[i]

        elif self.strategy == Strategy.RANDOM:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                n_missing = np.sum(mask)
                if n_missing > 0:
                    rng = np.random.default_rng(self.random_state)
                    random_values = rng.choice(
                        self.random_params_[i],
                        size=n_missing,
                        replace=True
                    )
                    X_imputed[mask, i] = random_values

        if self.add_indicator:
            indicators = np.array([self.indicators_[i]
                                  for i in range(X.shape[1])]).T
            X_imputed = np.hstack([X_imputed, indicators.astype(int)])

        if is_1d and not self.add_indicator:
            X_imputed = X_imputed.ravel()

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
