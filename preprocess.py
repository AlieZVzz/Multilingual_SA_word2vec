from Config import *
from torch.utils.data import DataLoader
import torch
import collections

# import d2l.torch as d2l



class TokenEmbedding:
    """Token Embedding."""

    def __init__(self, vec_dir):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(vec_dir)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, vec_dir):
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(vec_dir, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx)
            for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_path, num_steps):
        self.train_path = train_path
        self.x_train, self.y_train, self.x_validation, self.y_validation = [], [], [], []
        with open(train_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                data = line.strip().split('\t')
                self.y_train.append(int(data[0]))
                self.x_train.append(data[1].strip().split())
        self.vocab = Vocab(self.x_train, min_freq=5)
        self.train_features = torch.tensor([truncate_pad(
            self.vocab[line], num_steps, self.vocab['<pad>']) for line in self.x_train])
        self.y_train = torch.tensor(self.y_train)

    def __getitem__(self, idx):
        data = (self.train_features[idx], self.y_train[idx])
        return data

    def __len__(self):
        return len(self.x_train)

    def __vocab__(self):
        return self.vocab


