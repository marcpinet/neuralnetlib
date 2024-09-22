import numpy as np
import re

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encoded labels are binary vectors representing categorical values,
    with exactly one high (or "hot" = 1) bit indicating the presence of a specific category
    and all other bits low (or "cold" = 0)."""
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    labels = labels.astype(int)
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def apply_threshold(y_pred, threshold: float = 0.5):
    """Applies a threshold to the predictions. Typically used for binary classification."""
    return (y_pred > threshold).astype(int)


def im2col_2d(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Transform 4 dimensional images to 2 dimensional array.

    Args:
        input_data (np.ndarray): 4 dimensional input images (The number of images, The number of channels, Height, Width)
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int or tuple): the interval of stride
        pad (int or tuple): the interval of padding

    Returns:
        col (np.ndarray): 2 dimensional array

    """
    N, C, H, W = input_data.shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    # Make sure that the convolution can be executed
    assert (
                   H + 2 * pad_h - filter_h) % stride_h == 0, f'invalid parameters, (H + 2 * pad_h - filter_h) % stride_h != 0, got H={H}, pad_h={pad_h}, filter_h={filter_h}, stride_h={stride_h}'
    assert (
                   W + 2 * pad_w - filter_w) % stride_w == 0, f'invalid parameters, (W + 2 * pad_w - filter_w) % stride_w != 0, got W={W}, pad_w={pad_w}, filter_w={filter_w}, stride_w={stride_w}'

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            col[:, :, y, x, :, :] = padded_input[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def im2col_1d(input_data, filter_size, stride=1, pad=0):
    """
    Transform 3 dimensional images to 2 dimensional array.

    Args:
        input_data (np.ndarray): 3 dimensional input images (The number of images, The number of channels, Length)
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        col (np.ndarray): 2 dimensional array

    """
    N, C, L = input_data.shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad, pad)), mode='constant')

    col = np.zeros((N, C, filter_size, out_l))

    for y in range(filter_size):
        y_max = y + stride * out_l
        col[:, :, y, :] = padded_input[:, :, y:y_max:stride]

    col = col.transpose(0, 3, 1, 2).reshape(N * out_l, -1)
    return col


def col2im_1d(col, input_shape, filter_size, stride=1, pad=0):
    """
    Inverse of im2col_1d.

    Args:
        col (np.ndarray): 2 dimensional array
        input_shape (tuple): the shape of original input images
        filter_size (int): size of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        image (np.ndarray): original images

    """
    N, C, L = input_shape

    out_l = (L + 2 * pad - filter_size) // stride + 1

    col = col.reshape(N, out_l, C, filter_size).transpose(0, 2, 3, 1)

    image = np.zeros((N, C, L + 2 * pad + stride - 1))

    for y in range(filter_size):
        y_max = y + stride * out_l
        image[:, :, y:y_max:stride] += col[:, :, y, :]

    return image[:, :, pad:L + pad]


def col2im_2d(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Inverse of im2col.

    Args:
        col (np.ndarray): 2 dimensional array
        input_shape (tuple): the shape of original input images
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int or tuple): the interval of stride
        pad (int or tuple): the interval of padding

    Returns:
        image (np.ndarray): original images

    """
    N, C, H, W = input_shape

    if isinstance(pad, int):
        pad_h, pad_w = pad, pad
    else:
        pad_h, pad_w = pad

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    # Make sure that the convolution can be executed
    assert (
                   H + 2 * pad_h - filter_h) % stride_h == 0, f'invalid parameters, (H + 2 * pad_h - filter_h) % stride_h != 0, got H={H}, pad_h={pad_h}, filter_h={filter_h}, stride_h={stride_h}'
    assert (
                   W + 2 * pad_w - filter_w) % stride_w == 0, f'invalid parameters, (W + 2 * pad_w - filter_w) % stride_w != 0, got W={W}, pad_w={pad_w}, filter_w={filter_w}, stride_w={stride_w}'

    out_h = (H + 2 * pad_h - filter_h) // stride_h + 1
    out_w = (W + 2 * pad_w - filter_w) // stride_w + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    image = np.zeros((N, C, H + 2 * pad_h + stride_h - 1, W + 2 * pad_w + stride_w - 1))

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            image[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    return image[:, :, pad_h:H + pad_h, pad_w:W + pad_w]


def pad_sequences(sequences: np.ndarray, max_length: int, padding: str = 'pre', truncating: str = 'pre') -> np.ndarray:
    """Pads sequences to the same length.

    Args:
        sequences (np.ndarray): List of sequences.
        max_length (int): Maximum length of sequences.
        padding (str): 'pre' or 'post', pad either before or after each sequence.
        truncating (str): 'pre' or 'post', remove values from sequences larger than max_length, either at the beginning or at the end of the sequences.

    Returns:
        np.ndarray: Padded sequences.
    """
    padded_sequences = np.zeros((len(sequences), max_length))
    for i, sequence in enumerate(sequences):
        if len(sequence) > max_length:
            if truncating == 'pre':
                sequence = sequence[-max_length:]
            else:
                sequence = sequence[:max_length]
        if padding == 'pre':
            padded_sequences[i, -len(sequence):] = sequence
        else:
            padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_

    def transform(self, X):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * self.scale_ + self.min_


class PCA:
    def __init__(self, n_components: int, random_state: int = None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None
        self.mean = None
        self.input_shape = None

    def fit(self, X: np.ndarray):
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


class Tokenizer:
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None):
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.char_level = char_level
        self.oov_token = oov_token
        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}
        self.word_docs = {}
        self.document_count = 0

    def fit_on_texts(self, texts):
        for text in texts:
            self.document_count += 1
            if self.char_level:
                seq = text
            else:
                seq = text.split(self.split) if isinstance(text, str) else text
            for w in seq:
                if self.lower:
                    w = w.lower()
                if w in self.filters:
                    continue
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        
        # Note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        if self.num_words is not None:
            self.word_index = dict(list(self.word_index.items())[:self.num_words])

        self.index_word = dict((c, w) for w, c in self.word_index.items())

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        for text in texts:
            if self.char_level:
                seq = text
            else:
                seq = text.split(self.split) if isinstance(text, str) else text
            vect = []
            for w in seq:
                if self.lower:
                    w = w.lower()
                i = self.word_index.get(w)
                if i is not None:
                    if self.num_words and i >= self.num_words:
                        if self.oov_token is not None:
                            vect.append(self.word_index.get(self.oov_token))
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(self.word_index.get(self.oov_token))
            yield vect

    def sequences_to_texts(self, sequences):
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.oov_token)
            if self.char_level:
                yield ''.join(vect)
            else:
                yield ' '.join(vect)

    def get_config(self):
        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'char_level': self.char_level,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
        }
        

class CountVectorizer:
    def __init__(self, lowercase=True, token_pattern=r'(?u)\b\w\w+\b', 
                 max_df=1.0, min_df=1, max_features=None):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self.document_count_ = 0

    def _tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return re.findall(self.token_pattern, text)

    def fit(self, raw_documents):
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
            terms = sorted(terms, key=lambda t: term_freq[t], reverse=True)[:self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(terms))}

        return self

    def transform(self, raw_documents):
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        X = np.zeros((len(raw_documents), len(self.vocabulary_)), dtype=int)

        for doc_idx, doc in enumerate(raw_documents):
            for term in self._tokenize(doc):
                if term in self.vocabulary_:
                    X[doc_idx, self.vocabulary_[term]] += 1

        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self):
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get))

    def get_vocabulary(self):
        return dict(sorted(self.vocabulary_.items(), key=lambda x: x[1]))