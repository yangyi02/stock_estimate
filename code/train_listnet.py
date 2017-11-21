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
        self.fc4 = nn.Linear(num_hidden, num_hidden)
        self.bn4 = nn.BatchNorm1d(num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.bn5 = nn.BatchNorm1d(num_hidden)
        self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x1):
        x1 = self.bn0(x1)
        x1 = F.relu(self.bn1(self.fc1(x1)))
        x1 = F.relu(self.bn2(self.fc2(x1)))
        x1 = F.relu(self.bn3(self.fc3(x1)))
        x1 = F.relu(self.bn4(self.fc4(x1)))
        x1 = F.relu(self.bn5(self.fc5(x1)))
        y = self.fc(x1) 
        return y

    @staticmethod
    def get_next_batch(feature, score):
        sort_idx=score.argsort()
        x1 = feature[sort_idx,:]
        t=10*score[sort_idx]
        t=t[:,np.newaxis]
        return x1,t

def listwise_loss(y,gt):
    y_exp=torch.exp(y)
    y_exp_sum=torch.sum(y_exp,dim=0)
    gt_exp=torch.exp(gt)
    gt_exp_sum=torch.sum(gt_exp,dim=0)
    new_y=torch.div(y_exp,y_exp_sum)
    new_gt=torch.div(gt_exp,gt_exp_sum)
    log_new_y=torch.log(new_y)
    loss=-torch.mean(torch.mul(new_gt,log_new_y))
    return loss

def train(feature, score):
    model = BaseNet(feature.shape[1], 2048)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc_all = []
    for epoch in range(2000):
        optimizer.zero_grad()
        x1,gt0 = model.get_next_batch(feature, score)
        x1 = Variable(torch.from_numpy(x1).float())
        gt = Variable(torch.from_numpy(gt0).float())
        if torch.cuda.is_available():
            x1,gt= x1.cuda(),gt.cuda()
        y= model(x1)
        a=y.cpu().data.numpy()
        a=np.squeeze(a)
        gt0=np.squeeze(gt0)

        idx_pre=np.argsort(a)
        idx_gt =np.argsort(gt0)
        
        print "predicted top 100:"
        print idx_pre[0:100]

        print "gt top 100:"
        print idx_gt[0:100]
        print "the number of wrong location: %d"%np.count_nonzero(idx_pre-idx_gt)
   
        
        loss = listwise_loss(y, gt)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        print('epoch %d: loss %.4f' % (epoch, loss.data[0]))


def main():
    feature = []
    score = []
    i=2
    with open('/home/ccs/data/all/Quant_Datas_5.0/'+str(i)+'.csv', 'r') as f:
        a = f.readlines()
    with open('/home/ccs/data/all/id_scores/ID_'+str(i)+'.csv', 'r') as f2:
        b = f2.readlines()
    for item in a:
        line = item.strip().split(',')
        feature.append(line[0:])
    for item in b:
        line = item.strip().split(',')
        score.append(line[1])

    score = np.array(score).astype(np.float)
    feature = np.array(feature).astype(np.float)
    print(score.shape, feature.shape)

    train(feature, score)


if __name__ == '__main__':
    main()
