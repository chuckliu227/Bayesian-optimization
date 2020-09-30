import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from time import time
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

'''
基于pytorch实现，首先利用pytorch搭建基本的神经网络结构，然后通过调用GeneraterNet函数传入具体的网络参数
如此便可以将神经网络的训练train函数作为贝叶斯优化的目标函数
l,k,lr分别对应隐含层层数，隐含层节点数，学习率
其中多加了一个ratio，当l>1时，每一个隐含层的节点个数都是上一层节点个数的ratio倍，也就是节点数会逐渐下降，更贴近实际情况一点。
'''

class myerror(Exception):
    '''跳出循环用的自定义异常'''
    pass

class MySet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return self.data.shape[0]

class Net(nn.Module):
    def __init__(self, l, k, ratio):
        super(Net, self).__init__()
        self.fc = GenerateNet(l=l, k=k, ratio=ratio)

    def forward(self, x):
        x = self.fc(x)
        return x


def GenerateNet(l, k, ratio, input=4, out=4):
    lst = [nn.Linear(in_features=input, out_features=k, bias=True), nn.ReLU()]
    for i in range(l):
        lst.append(nn.Linear(int(k*ratio**i), int(k*ratio**(i+1)), bias=True))
        lst.append(nn.ReLU())
    lst[-1] = nn.Sigmoid()
    lst.append(nn.Linear(int(k*ratio**l), out_features=out, bias=True))
    return nn.Sequential(*lst)

def train(l, lr, k=7, ratio=0.95):
    net = Net(l=int(l), k=int(k), ratio=ratio)
    epocs = 1
    optim = t.optim.Adam(params=net.parameters(), lr=lr)
    net.train()
    for i in range(epocs):
        for data, label in train_loader:
            optim.zero_grad()
#            data, label = data.cuda(), label.cuda()
            output = net(data)
            loss = nn.CrossEntropyLoss()(output, label.long())
            loss.backward()
            optim.step()
    net.eval()
    predlis = []
    reallis = []
    for data, label in test_loader:
#        data, label = data.cuda(), label.cuda()
        output = net(data)
        pred = output.argmax(dim=1, keepdim=True)
        predlis.append(np.array(pred.cpu()).squeeze())
        reallis.append(np.array(label.cpu()).squeeze())
    predlis, reallis = np.concatenate(predlis), np.concatenate(reallis)
    return f1_score(predlis, reallis, average='micro')

def bp_opt(param, iter=20):
    t1 = time()
    svm_bo = BayesianOptimization(train, param)
    svm_bo.maximize(init_points=10, n_iter=20)
    res = svm_bo.max
    print('l:{}\tlr:{}\tf1:{}'.format(int(res['params']['l']), round(res['params']['lr'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))

    t2 = time()
    resarr = np.zeros(3)
    i = 0
    try:
        for l in range(param['l'][0], param['l'][1]+1):
            for lr in np.arange(param['lr'][0], param['lr'][1]+1, 0.005):
                res = train(l, lr)
                if res >= resarr[2]:
                    resarr = np.array([l, lr, res])
                    i = 0
                else:
                    i += 1
                if i >= iter:
                    raise myerror()
    except myerror:
        pass
    print('l:{}\tlr:{}\tf1:{}'.format(int(resarr[0]), round(resarr[1], 3), round(resarr[2], 3)))
    print('Grid Search:', round(time()-t2, 3))


dataset = load_iris()
data, label = dataset['data'], dataset['target']
mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
data = (data - mmin)/(mmax - mmin)
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, stratify=label)
trainX, testX, trainY, testY = trainX.astype(np.float32), testX.astype(np.float32), trainY.astype(np.long), testY.astype(np.long)
TrainData = MySet(data=trainX, label=trainY)
TestData = MySet(data=testX, label=testY)
train_loader = DataLoader(TrainData, batch_size=16, shuffle=True)
test_loader = DataLoader(TestData, batch_size=16, shuffle=True)
param = {'l':(1, 3), 'lr':(0.01, 0.5)}
bp_opt(param)

