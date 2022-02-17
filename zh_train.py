import torch
import d2l.torch as d2l
from preprocess import load_data, TokenEmbedding
from Config import *
import torch.nn as nn
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score
import numpy as np
import os
from torch.autograd import Variable
from SAmodel import BiRNN


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def evaluate(model, validation_data):
    model.eval()
    eval_loss, eval_acc = 0, 0
    for i, data in enumerate(validation_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.data.item()
        _, pred = torch.max(out, 1)
        num_acc = (pred == y).sum()
        eval_acc += num_acc.data.item()

    logger.info('validation loss is:{:.6f},validation acc is:{:.6f}'
                .format(eval_loss / (len(validation_data) * Config.batch_size),
                        eval_acc / (len(validation_data) * Config.batch_size)))
    return eval_acc, eval_loss


def train_and_eval(train_data, validation_data, criterion):
    best_acc = 0
    best_model = None
    logger.info('start training!')
    for epoch in range(Config.epoch):
        train_loss, train_acc = 0, 0
        true_label = []
        pred_label = []
        model.train()
        for i, data in tqdm(enumerate(train_data), total=len(train_data)):
            x, y = data
            x, y = Variable(x).cuda(), Variable(y).cuda()

            # forward
            out = model(x)

            loss = criterion(out, y)
            pred = out.argmax(axis=1)
            # _, pre = torch.max(out, 1)
            num_acc = (pred.type(y.dtype) == y).type(y.dtype).sum()
            train_acc += num_acc.data.item()
            true_label = np.append(true_label, y.tolist())
            pred_label = np.append(pred_label, pred.tolist())
            # backward
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            train_loss += loss.data.item()
            acc = accuracy_score(true_label, pred_label)
            train_dict['train_acc'].append(acc)
        # logger
        logger.info('epoch [{}]: train loss is:{:.6f},train acc is:{:.6f}'
                    .format(epoch + 1, train_loss / (len(train_data) * Config.batch_size),
                            train_acc / (len(train_data) * Config.batch_size)))
        # 测试
        eval_acc, eval_loss = evaluate(model, validation_data)
        train_dict['validation_acc'].append(eval_acc / (len(validation_data) * Config.batch_size))
        train_dict['validation_loss'].append(eval_loss / (len(validation_data) * Config.batch_size))
        # 保存
        if best_acc < (eval_acc / (len(validation_data) * Config.batch_size)):
            best_acc = eval_acc / (len(validation_data) * Config.batch_size)
            best_model = model
            logger.info('best model is changed, best acc is {:.6f}'.format(best_acc))

    logger.info('finish training!')
    torch.save(best_model.state_dict(), 'model/zh_BiLSTM.pth')
    np.save(os.path.join(save_path, 'train_acc.npy'), np.array(train_dict['train_acc']))
    np.save(os.path.join(save_path, 'val_acc.npy'), np.array(train_dict['validation_acc']))
    np.save(os.path.join(save_path, 'val_loss.npy'), np.array(train_dict['validation_loss']))


if __name__ == '__main__':
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
    train_iter, val_iter, vocab = load_data(x_train, y_train, batch_size=64)
    train_dict = {'train_acc': [], 'train_loss': [], 'validation_acc': [], 'validation_loss': []}
    logger = get_logger('log/zh_BiLSTM.log')
    save_path = 'save/zh_BiLSTM'
    model = BiRNN(len(vocab), Config.embedding_dim, Config.hidden_dim, Config.layer_size).cuda()
    model.apply(init_weights)
    glove_embedding = TokenEmbedding(vec_dir='data/sgns.sogounews.bigram-char')
    embeds = glove_embedding[vocab.idx_to_token]
    model.embedding.weight.data.copy_(embeds)
    model.embedding.weight.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimzier = torch.optim.Adam(model.parameters(), lr=Config.lr)
    train_and_eval(train_iter, val_iter, criterion)
