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


def GenerateNet(l, k, ratio, input=6):
    lst = [nn.Linear(in_features=input, out_features=k, bias=True), nn.ReLU()]
    for i in range(l):
        lst.append(nn.Linear(int(k*ratio**i), int(k*ratio**(i+1)), bias=True))
        lst.append(nn.ReLU())
    lst[-1] = nn.Sigmoid()
    lst.append(nn.Linear(int(k*ratio**l), out_features=4, bias=True))
    return nn.Sequential(*lst)

def train(l=1, k=10, lr=0.2, ratio=0.95):
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

def bp_opt(param, iter=15):
    t1 = time()
    svm_bo = BayesianOptimization(train, param)
    svm_bo.maximize(init_points=5, n_iter=10)
    res = svm_bo.max
    print('l:{}\tk:{}\tlr:{}\tf1:{}'.format(int(res['params']['l']), int(res['params']['k']), round(res['params']['lr'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))

    t2 = time()
    resarr = np.zeros(4)
    i = 0
    try:
        for l in range(param['l'][0], param['l'][1]+1):
            for k in np.arange(param['k'][0], param['k'][1]+1, 1):
                for lr in np.arange(param['lr'][0], param['lr'][1]+1, 0.005):
                    res = train(l, k, lr)
                    if res >= resarr[3]:
                        resarr = np.array([l, k, lr, res])
                        i = 0
                    else:
                        i += 1
                    if i >= iter:
                        raise myerror()
    except myerror:
        pass
    print('l:{}\tk:{}\tlr:{}\tf1:{}'.format(int(resarr[0]), int(resarr[1]), round(resarr[2], 3), round(resarr[3], 3)))
    print('Grid Search:', round(time()-t2, 3))

def bp_opt_lr(param, iter=15):
    t1 = time()
    svm_bo = BayesianOptimization(train, param)
    svm_bo.maximize(init_points=5, n_iter=10)
    res = svm_bo.max
    print('lr:{}\tf1:{}'.format(round(res['params']['lr'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))

    t2 = time()
    resarr = np.zeros(2)
    i = 0
    try:
        for lr in np.arange(param['lr'][0], param['lr'][1]+1, 0.005):
            res = train(l=1, k=10, lr=lr)
            if res >= resarr[1]:
                resarr = np.array([lr, res])
                i = 0
            else:
                i += 1
            if i >= iter:
                raise myerror()
    except myerror:
        pass
    print('lr:{}\tf1:{}'.format(round(resarr[0], 3), round(resarr[1], 3)))
    print('Grid Search:', round(time()-t2, 3))


data = pd.read_csv(r'cars.data', header=None)
buying      = {'vhigh':0,   'high':1,   'med':2,    'low':3}
maint       = {'vhigh':0,   'high':1,   'med':2,    'low':3}
doors       = {'2':0,       '3':1,      '4':2,      '5more':3}
persons     = {'2':0,       '4':1,      'more':2}
lug_boot    = {'small':0,   'med':1,    'big':2}
safety      = {'low':0,     'med':1,    'high':2}
label       = {'unacc':0,   'acc':1,    'good':2,   'vgood':3}
dictlis = [buying, maint, doors, persons, lug_boot, safety, label]
for idx, item in enumerate(data.iloc):
    for i in range(7):
        data[i][idx] = dictlis[i][data[i][idx]]
file = np.array(data)
data, label = file[:, :-1], file[:, -1]
mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
data = (data - mmin)/(mmax - mmin)
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, stratify=label)
trainX, testX, trainY, testY = trainX.astype(np.float32), testX.astype(np.float32), trainY.astype(np.long), testY.astype(np.long)
TrainData = MySet(data=trainX, label=trainY)
TestData = MySet(data=testX, label=testY)
train_loader = DataLoader(TrainData, batch_size=16, shuffle=True)
test_loader = DataLoader(TestData, batch_size=16, shuffle=True)

param = {'l':(1, 3), 'k':(6, 12), 'lr':(0.001, 0.5)}
bp_opt(param)

param = {'lr':(0.001, 0.5)}
bp_opt_lr(param)