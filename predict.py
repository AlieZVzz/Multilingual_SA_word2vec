import torch.nn as nn
import torch
from Config import *
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sentiment_analysis_model import BiRNN
import numpy as np
import d2l.torch as d2l
from preprocess import *


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


test_data = Dataset(Config.validation_path, num_steps=500)
test_iter = DataLoader(test_data, batch_size=64)
state_dict = torch.load('model/eng_BiLSTM.pth', map_location='cuda')
model = BiRNN(49221, embed_size=Config.embed_size, num_hiddens=Config.num_hiddens, num_layers=Config.num_layers)
model.load_state_dict(state_dict=state_dict)
criterion = nn.CrossEntropyLoss()
sk_loss, eval_acc, eval_loss, matrix = evaluate(model, test_iter, criterion=criterion)
print(matrix)
