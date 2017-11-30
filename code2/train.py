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
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.bn3 = nn.BatchNorm1d(num_hidden)
        self.fc4 = nn.Linear(num_hidden * 2, num_hidden)
        self.bn4 = nn.BatchNorm1d(num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.bn5 = nn.BatchNorm1d(num_hidden)
        self.fc = nn.Linear(num_hidden, 3)

    def forward(self, x1, x2):
        x1 = self.bn0(x1)
        x1 = F.relu(self.bn1(self.fc1(x1)))
        x1 = F.relu(self.bn2(self.fc2(x1)))
        x1 = F.relu(self.bn3(self.fc3(x1)))
        x2 = self.bn0(x2)
        x2 = F.relu(self.bn1(self.fc1(x2)))
        x2 = F.relu(self.bn2(self.fc2(x2)))
        x2 = F.relu(self.bn3(self.fc3(x2)))
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        y = self.fc(x)
        return y


def get_next_batch(stock_price_ratios, batch_size=128):
    """
    Prepare training data with the given stock price ratios
    The goal is not to predict the stock price (ratio) value but to find which stock will be
    the most increased one in the next time step
    We use pairwise ranking to compare between two stocks
    Features are the stock price ratio at previous 30 time steps
    Outputs are the comparison between two stocks at next stock price ratio
    """
    day_id = np.random.randint(len(stock_price_ratios))
    idx1 = np.random.randint(stock_price_ratios[day_id].shape[0], size=batch_size)
    idx2 = np.random.randint(stock_price_ratios[day_id].shape[0], size=batch_size)
    time = np.random.randint(stock_price_ratios[day_id].shape[1]-30)
    x1 = stock_price_ratios[day_id][idx1, time:time+30]
    x2 = stock_price_ratios[day_id][idx2, time:time+30]
    ratio_diff = stock_price_ratios[day_id][idx1, time+30] - stock_price_ratios[day_id][idx2, time+30]
    y = np.zeros_like(ratio_diff)
    y[ratio_diff > 0] = 0
    y[ratio_diff < 0] = 1
    y[ratio_diff == 0] = 2
    prop = np.zeros(3)
    for i in range(3):
        prop[i] = np.sum(y == i).astype(np.float) / y.shape[0]
    print('proption: 0: %.2f, 1: %.2f, 2: %.2f' % (prop[0], prop[1], prop[2]))
    return x1, x2, y


def train(stock_price_ratios, num_epoch=2000):
    """
    Train with the given stock prices
    The goal is not to predict the stock price value but to find which stock will be
    the most increased one in the next time step
    We use pairwise ranking to compare between two stocks
    Features are the stock price ratio at previous 30 time steps
    Outputs are the comparison between two stocks at next stock price ratio
    """
    model = BaseNet(30, 128)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc_all = []
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        x1, x2, gt = get_next_batch(stock_price_ratios)
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
        print('train epoch %d: loss %.4f, acc: %.4f' % (epoch, loss.data[0], ave_acc))
    return model


def test(stock_price_ratios, model, num_epoch=10):
    """
    Test with the new stock prices
    The goal is not to predict the stock price value but to find which stock will be
    the most increased one in the next time step
    We use pairwise ranking to compare between two stocks
    Features are the stock price ratio at previous 30 time steps
    Outputs are the comparison between two stocks at next stock price ratio
    """
    acc_all = []
    for epoch in range(num_epoch):
        x1, x2, gt = get_next_batch(stock_price_ratios)
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
        if epoch > 100:
            acc_all.pop(0)
        ave_acc = sum(acc_all) / float(len(acc_all))
        print('test epoch %d: loss %.4f, acc: %.4f' % (epoch, loss.data[0], ave_acc))


def prepare_data(lines):
    """
    Change the original data format to deep learning prediction data format

    The original data format:
              time1, time2, time3, ...
    stock1    price1, price2, price3, ...
    stock2    price1, price2, prcie3, ...

    The deep learning data format:
              time1, time2, time3, ...
    stock1    price_ratio1, price_ratio2, price_ratio3, ...
    stock2    price_ratio1, price_ratio2, price_ratio3, ...
    """
    stock_price_ratios = []
    for item in lines:
        data = []
        for current_line in item:
            line = current_line.strip().split(',')
            data.append(line)
        data = np.array(data)
        data = data[1:, 1:]
        data = data.astype(np.float)
        print(data)
        print(data.shape)
        data = np.log(data[:, 1:] / data[:, :-1])
        print(data)
        print(data.shape)
        stock_price_ratios.append(data)
    return stock_price_ratios


def main():
    stock_file_list = ['Nov20_5m.csv', 'Nov21_5m.csv']
    total_lines = []
    for stock_file in stock_file_list:
        with open(stock_file, 'r') as f:
            lines = f.readlines()
        total_lines.append(lines)
    stock_price_ratios = prepare_data(total_lines)
    get_next_batch(stock_price_ratios)
    model = train(stock_price_ratios)
    test(stock_price_ratios, model)


if __name__ == '__main__':
    main()
