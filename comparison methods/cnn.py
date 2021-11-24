# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:44:22 2021

https://blog.csdn.net/qq_39064418/article/details/120909287

cnn

@author: wangyin
"""
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio 
from sklearn import metrics as met

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#参数
trainstep=20
predstep=10
epochs = 100
noise = 0
# 初始化预测的数组和对比的数组
ypred=[]
yref=[]

#datafile='kssolution256x0-200t01.mat'
#datafile='brusselator.mat'
datafile='sh_data.mat'
datamat=scio.loadmat(datafile)
uu=datamat['uu'] #第一维是x 第二维是t

uu = torch.FloatTensor(uu)
#定义一些误差评价函数
def R2func(fi_list,yi_list):
    fi_list=np.array(fi_list)
    yi_list=np.array(yi_list)
    y_mean = np.mean(yi_list)
    ss_tot = np.sum((yi_list-y_mean)**2)
    ss_err = np.sum((yi_list-fi_list)**2)
    r2 = 1 - (ss_err/ss_tot)
    return r2
#https://blog.csdn.net/weixin_38100489/article/details/78175928
def adjusted_R2func(predinput,realinput,dimension):
    predinput=np.array(predinput)
    realinput=np.array(realinput)
    
    n=np.size(predinput)
    n=256
    p=dimension
    rr=R2func(predinput,realinput)
    adjust_rr=1-(1-rr)*(n-1)/(n-p-1)
    
    return adjust_rr

timestart=1

#数据集和目标值赋值，dataset为数据，i为起始位置，look_back为以几行数据为特征维度数量
#predict_forward为训练的预测长度
shape=np.shape(uu)
shape1=shape[0]
shape2=shape[1]

uu_noise=uu+noise*np.random.uniform(low=-1, high=1, size=[shape1,shape2])

def creat_dataset(dataset,i,look_back,predict_forward):
    out = []
    shape12=np.shape(dataset)
    shape12=shape12[0]
    for j in range(shape1):
        data_x=dataset[j,i:i+look_back]
        data_y=dataset[j,i+look_back:i+look_back+predict_forward]
        out.append((data_x, data_y))
    return out #转为ndarray数据

#######################################################开始cnn的代码
class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1,64,kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*(trainstep-1),50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        # 该模型的网络结构为 一维卷积层 -> Relu层 -> Flatten -> 全连接层1 -> 全连接层2 
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
while timestart<shape2:
    

    print(timestart)

    #输出训练数据
    train_data = creat_dataset(uu_noise,timestart,trainstep,1)
    
    
    
    for data_x, data_y in train_data:
        data_x.reshape(1,1,-1)
    data_x.reshape(1,1,-1)
    
    
    
    torch.manual_seed(101)
    model =CNNnetwork()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
      
    
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        
        for data_x, data_y in train_data:
            
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
    
            # 注意这里要对样本进行reshape，转换成conv1d的input size（batch size, channel, series length）
            y_pred = model(data_x.reshape(1,1,-1))
            loss = criterion(y_pred, data_y)
            loss.backward()
            optimizer.step()
            
    #    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
        
    
    
    
    
    
    # 设置成eval模式
    model.eval()
    # 进行预测
    
    # 循环的每一步表示向时间序列向后滑动一格
    for j in range(shape1):
        pred_input = uu_noise[j,timestart+1:timestart+trainstep+1].tolist()
        for q in range(predstep):
            #输出对比的数据
            yref.append(uu[j,timestart+trainstep+1+q:timestart+trainstep+1+1+q])
            seq = torch.FloatTensor(pred_input[-trainstep:])
            with torch.no_grad():
                pred_input.append(model(seq.reshape(1,1,-1)).item())
                #输出预测的数据
                ypred.append(model(seq.reshape(1,1,-1)).item())
                
    result_pred=ypred
    result_real=yref
    
    #计算误差

    npreal=np.array(yref)
    nppred=np.array(ypred)
    MSE=met.mean_squared_error(ypred,yref)
    MAE=met.mean_absolute_error(ypred,yref)
    RMSE=met.mean_squared_error(ypred,yref)**0.5
    MAD=met.median_absolute_error(ypred,yref)
    R2=R2func(nppred,npreal)
    adjR2=adjusted_R2func(ypred,yref,trainstep)
    index=np.argwhere(npreal==0)
    npreal=np.delete(npreal,index)
    nppred=np.delete(nppred,index)
    MAPE=np.mean(np.abs((nppred-npreal)/npreal))
    print('MSE',MSE)
    print('MAE',MAE)
    print('RMSE',RMSE)
    print('MAD',MAD)
    print('MAPE',MAPE)
    print('R2',R2)
    print('adjR2',adjR2)

    timestart=timestart+predstep
