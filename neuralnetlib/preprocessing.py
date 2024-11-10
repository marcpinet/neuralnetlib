
import re
import random
import numpy as np
from enum import Enum, auto
from collections import defaultdict
from collections.abc import Generator


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


def apply_threshold(y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Applies a threshold to the predictions. Typically used for binary classification."""
    return (y_pred > threshold).astype(int)


def im2col_2d(input_data: np.ndarray, filter_h: int, filter_w: int, stride: int | tuple[int, int] = 1, pad: int | tuple[int, int] = 0) -> np.ndarray:
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


def im2col_1d(input_data: np.ndarray, filter_size: int, stride: int = 1, pad: int = 0) -> np.ndarray:
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


def col2im_1d(col: np.ndarray, input_shape: tuple[int, int, int], filter_size: int, stride: int = 1, pad: int = 0) -> np.ndarray:
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


def col2im_2d(col: np.ndarray, input_shape: tuple[int, int, int, int], filter_h: int, filter_w: int, stride: int | tuple[int, int] = 1, pad: int | tuple[int, int] = 0) -> np.ndarray:
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

    def fit(self, X: np.ndarray) -> None:
        self.min_ = np.min(X, axis=0)
        self.scale_ = np.max(X, axis=0) - self.min_

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
    def __init__(self,  num_words: int | None = None, filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower: bool = True, split: str = ' ', char_level: bool = False, oov_token: str | None = None) -> None:
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

    def fit_on_texts(self, texts: list[str]) -> None:
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

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts: list[str]) -> Generator[list[int], None, None]:
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

    def sequences_to_texts(self, sequences: list[list[int]]) -> list[str]:
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences: list[list[int]]) -> Generator[str, None, None]:
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

    def get_config(self) -> dict:
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
    def __init__(self, lowercase: bool = True, token_pattern: str = r'(?u)\b\w\w+\b', max_df: float | int = 1.0, min_df: float | int = 1, max_features: int | None = None) -> None:
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
            terms = sorted(terms, key=lambda t: term_freq[t], reverse=True)[:self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(terms))}

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
    


class TokenType(Enum):
    CHAR = auto()
    WORD = auto()

class NGram:
    def __init__(self, 
                 n: int = 3, 
                 token_type: TokenType = TokenType.CHAR,
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
        if self.token_type == TokenType.CHAR:
            return list(text)
        return text.split(self.separator)
        
    def _join_tokens(self, tokens: list[str]) -> str:
        if self.token_type == TokenType.CHAR:
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
        if self.token_type == TokenType.CHAR:
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
                context = tuple(current[-(self.n-1):])
                
                if context not in self.ngrams:
                    if (self.token_type == TokenType.WORD and 
                        current[-1] in self.transitions):
                        next_token = random.choice(self.transitions[current[-1]])
                        current.append(next_token)
                        continue
                    break
                    
                next_token = random.choice(self.ngrams[context])
                current.append(next_token)
                
                if next_token == self.end_token:
                    sequence = current[(self.n-1):-1]
                    if len(sequence) >= min_length:
                        if max_length is None or len(sequence) <= max_length:
                            result = self._join_tokens(sequence)
                            if self.token_type == TokenType.WORD:
                                result = result.capitalize()
                            return result
                    break
                
                if max_length and len(current) - (self.n-1) > max_length:
                    break
                    
                if (self.token_type == TokenType.WORD and 
                    random.random() < variability and 
                    current[-1] in self.transitions):
                    next_token = random.choice(self.transitions[current[-1]])
                    current.append(next_token)
        
        raise ValueError(f"Could not generate a sequence after {max_attempts} attempts.")
    
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
