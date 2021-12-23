from Config import *
import torch
import d2l.torch as d2l
import os

x_train = []
y_train = []
x_validation = []
y_validation = []
with open(Config.train_path, 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        data = line.strip().split('\t')
        y_train.append(int(data[0]))
        x_train.append(data[1].strip().split())

with open(Config.validation_path, 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        data = line.strip().split('\t')
        y_validation.append(int(data[0]))
        x_validation.append(data[1].strip().split())


class TokenEmbedding:
    """Token Embedding."""

    def __init__(self, data_dir):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(data_dir)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, data_dir):
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(os.path.join(data_dir, 'sgns.sogounews.bigram-char'), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx)
            for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def load_data(x_train, y_train, batch_size, num_steps=500):
    vocab = d2l.Vocab(x_train, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in x_train])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in x_validation])
    train_iter = d2l.load_array((train_features, torch.tensor(y_train)),
                                batch_size)
    validation_iter = d2l.load_array((test_features, torch.tensor(y_validation)),
                                     batch_size,
                                     is_train=False)
    return train_iter, validation_iter, vocab
