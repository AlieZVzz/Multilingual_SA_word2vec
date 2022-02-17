import torch.nn as nn
import torch
from Config import *
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from SAmodel import BiRNN
import numpy as np
import d2l.torch as d2l


def evaluate(model, test_data, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    true_label = []
    pred_label = []

    for i, data in enumerate(test_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x)
            y = Variable(y)
        out = model(x)
        loss = criterion(out, y)
        test_loss += loss.data.item()
        _, pre = torch.max(out, 1)
        true_label = np.append(true_label, y.tolist())
        pred_label = np.append(pred_label, pre.tolist())
        num_acc = (pre == y).sum()
        test_acc += num_acc.data.item()

    print('test loss is:{:.6f},test acc is:{:.6f}'
          .format(test_loss / (len(test_data) * Config.batch_size),
                  test_acc / (len(test_data) * Config.batch_size)))
    sk_acc = accuracy_score(true_label, pred_label)
    confusion_m = confusion_matrix(true_label, pred_label)
    return sk_acc, test_acc, test_loss, confusion_m


def load_test_data(test_path, batch_size, num_steps=500):
    x_test = []
    y_test = []
    with open(test_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_test.append(int(data[0]))
            x_test.append(data[2].strip().split())
    vocab = d2l.Vocab(x_test, min_freq=5)
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in x_test])
    test_iter = d2l.load_array((test_features, torch.tensor(y_test)), batch_size, is_train=False)
    return test_iter, vocab


test_data, vocab = load_test_data(Config.test_path, batch_size=64)
state_dict = torch.load('model/zh_BiLSTM.pth', map_location='cpu')
model = BiRNN(16176, embed_size=Config.embedding_dim, num_hiddens=Config.hidden_dim, num_layers=Config.layer_size)
model.load_state_dict(state_dict=state_dict)
criterion = nn.CrossEntropyLoss()
sk_loss, eval_acc, eval_loss, matrix = evaluate(model, test_data, criterion=criterion)
print(matrix)
