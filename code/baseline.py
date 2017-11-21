import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self, num_input, num_hidden):
        super(BaseNet, self).__init__()
        self.bn0 = nn.BatchNorm1d(num_input)
        self.fc = nn.Linear(num_input * 2, 3)

    def forward(self, x1, x2):
        x1 = self.bn0(x1)
        x2 = self.bn0(x2)
        x = torch.cat((x1, x2), 1)
        y = self.fc(x)
        return y

    @staticmethod
    def get_next_batch(feature, score, batch_size=512):
        idx1 = np.random.randint(score.shape[0], size=batch_size)
        idx2 = np.random.randint(score.shape[0], size=batch_size)
        x1 = feature[idx1, :]
        x2 = feature[idx2, :]
        score_diff = score[idx1] - score[idx2]
        y = np.zeros_like(score_diff)
        thresh = 0.5
        y[score_diff >= thresh] = 1
        y[np.abs(score_diff) < thresh] = 0
        y[score_diff <= -thresh] = 2
        prop = np.zeros(3)
        for i in range(3):
            prop[i] = np.sum(y == i).astype(np.float) / y.shape[0]
        print('proption: 0: %.2f, 1: %.2f, 2: %.2f' % (prop[0], prop[1], prop[2]))
        return x1, x2, y


def train(feature, score):
    model = BaseNet(feature.shape[1], 2048)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc_all = []
    for epoch in range(2000):
        optimizer.zero_grad()
        x1, x2, gt = model.get_next_batch(feature, score)
        x1 = Variable(torch.from_numpy(x1).float())
        x2 = Variable(torch.from_numpy(x2).float())
        gt = Variable(torch.from_numpy(gt).long())
        if torch.cuda.is_available():
            x1, x2, gt = x1.cuda(), x2.cuda(), gt.cuda()
        y = model(x1, x2)
        loss = F.cross_entropy(y, gt)
        pred = y.data.max(1)[1]
        acc = pred.eq(gt.data).cpu().sum() * 1.0 / gt.numel()
        acc_all.append(acc)
        loss.backward()
        optimizer.step()
        if epoch > 100:
            acc_all.pop(0)
        ave_acc = sum(acc_all) / float(len(acc_all))
        print('epoch %d: loss %.4f, acc: %.4f' % (epoch, loss.data[0], ave_acc))


def main():
    with open('../998.csv', 'r') as f:
        a = f.readlines()

    feature = []
    for item in a:
        line = item.strip().split(',')
        feature.append(line)

    with open('../id_score/ID_998.csv', 'r') as f:
        b = f.readlines()

    score = []
    for item in b:
        line = item.strip().split(',')
        score.append(line[1])

    score = np.array(score).astype(np.float)
    feature = np.array(feature).astype(np.float)
    print(score.shape, feature.shape)

    train(feature, score)


if __name__ == '__main__':
    main()
