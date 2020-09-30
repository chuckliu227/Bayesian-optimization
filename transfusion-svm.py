import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from time import time
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score

class myerror(Exception):
    pass

def svm_opt(c, scale):
    '''
    贝叶斯优化的目标函数，输入SVM的参数输出SVM的F1-score
    下面两个相同
    '''
    svc = SVC(C=c, gamma=scale)
    svc.fit(trainX, trainY)
    res = svc.predict(testX)
    return f1_score(res, testY, average='micro')

def svm_opt_c(c):
    svc = SVC(C=c)
    svc.fit(trainX, trainY)
    res = svc.predict(testX)
    return f1_score(res, testY, average='micro')

def svm_opt_gamma(scale):
    svc = SVC(gamma=scale)
    svc.fit(trainX, trainY)
    res = svc.predict(testX)
    return f1_score(res, testY, average='micro')

def Optimize_SVM(param, iter=20):
    # 贝叶斯优化
    t1 = time()
    svm_bo = BayesianOptimization(svm_opt, param)
    svm_bo.maximize(init_points=5, n_iter=5)
    res = svm_bo.max
    print('c:{}\tgamma:{}\tf1:{}'.format(round(res['params']['c'], 3), round(res['params']['scale'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))  

    # 全局搜索
    t2 = time()
    resarr = np.zeros(3)
    i = 0
    try:
        for c in np.arange(param['c'][0], param['c'][1], 0.2):
            for gamma in np.arange(param['scale'][0], param['scale'][1], 0.001):
                svc = SVC(C=c, gamma=gamma)
                svc.fit(trainX, trainY)
                res = f1_score(svc.predict(testX), testY, average='micro')
                # 记录最佳搜索结果
                if res >= resarr[2]:
                    resarr = np.array([c, gamma, res])
                    i = 0
                else:
                    i += 1
                # Grid-search 停止条件为最佳结果一定轮次不更新
                if i >= iter:
                    raise myerror()
    except myerror:
        pass
    print('c:{}\tgamma:{}\tf1:{}'.format(round(resarr[0], 3), round(resarr[1], 3), round(resarr[2], 3)))
    print('Grid Search:', round(time()-t2, 3))

def Optimize_c(param, iter=20):
    t1 = time()
    svm_bo = BayesianOptimization(svm_opt_c, param)
    svm_bo.maximize(init_points=5, n_iter=10)
    res = svm_bo.max
    print('c:{}\tf1:{}'.format(round(res['params']['c'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))
    t2 = time()
    resarr = np.zeros(2)
    i = 0
    try:
        for c in np.arange(param['c'][0], param['c'][1], 0.2):
            svc = SVC(C=c)
            svc.fit(trainX, trainY)
            res = f1_score(svc.predict(testX), testY, average='micro')
            if res >= resarr[1]:
                resarr = np.array([c, res])
                i = 0
            else:
                i += 1
            if i >= iter:
                raise myerror()
    except myerror:
        pass
    print('c:{}\tf1:{}'.format(round(resarr[0], 3), round(resarr[1], 3)))
    print('Grid Search:', round(time()-t2, 3))

def Optimize_gamma(param, iter=20):
    t1 = time()
    svm_bo = BayesianOptimization(svm_opt_gamma, param)
    svm_bo.maximize(init_points=5, n_iter=10)
    res = svm_bo.max
    print('gamma:{}\tf1:{}'.format(round(res['params']['scale'], 3), round(res['target'], 3)))
    print('Bayesian Optimization:', round(time()-t1, 3))

    t2 = time()
    resarr = np.zeros(2)
    i = 0
    try:
        for gamma in np.arange(param['scale'][0], param['scale'][1], 0.001):
            svc = SVC(gamma=gamma)
            svc.fit(trainX, trainY)
            res = f1_score(svc.predict(testX), testY, average='micro')
            if res >= resarr[1]:
                resarr = np.array([gamma, res])
                i = 0
            else:
                i += 1
            if i >= iter:
                raise myerror()
    except myerror:
        pass
    print('gamma:{}\tf1:{}'.format(round(resarr[0], 3), round(resarr[1], 3)))
    print('Grid Search:', round(time()-t2, 3))

# ----------------读取数据并制作献血数据集----------------
data = np.array(pd.read_csv(r'transfusion.data'))
mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
data = (data - mmin)/(mmax - mmin)
trainX, testX, trainY, testY = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, stratify=data[:, -1])


# ----------------2分类2参数----------------
param = {'c':(0.01, 3), 'scale':(0.001, 3)}
Optimize_SVM(param)


# ----------------2分类1参数----------------
param = {'c':(0.01, 3)}
Optimize_c(param)


# ----------------2分类1参数----------------
param = {'scale':(0.001, 3)}
Optimize_gamma(param)