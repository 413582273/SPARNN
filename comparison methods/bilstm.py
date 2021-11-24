# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:00:31 2021

@author: wangyin

https://blog.csdn.net/qq_37236745/article/details/107077024
"""
import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as met

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
    
    
    

#参数
trainstep=21
predstep=10

n_hidden = 5
n_class=trainstep

result_real=[]
result_pred=[]


# 读取mat文件
import scipy.io as scio

#datafile='kssolution256x0-200t01.mat'
#datafile='brusselator.mat'
datafile='sh_data.mat'
datamat=scio.loadmat(datafile)
uu=datamat['uu'] #第一维是x 第二维是t

#数据集和目标值赋值，dataset为数据，i为起始位置，look_back为以几行数据为特征维度数量
#predict_forward为训练的预测长度
def creat_dataset(dataset,i,look_back,predict_forward):
    data_x=[]
    data_y=[]
    data_x=dataset[:,i:i+look_back]
    data_y=dataset[:,i+look_back:i+look_back+predict_forward]
    return np.asarray(data_x), np.asarray(data_y) #转为ndarray数据

for q in range(300):
    print(q)
    #得到训练数据集
    dataX, dataY = creat_dataset(uu,q*predstep,trainstep,predstep)  #第一维是x 第二维是t
    x_train = dataX #训练数据
    y_train = dataY #训练数据目标值
    
    x_train = x_train.reshape(-1, 1, trainstep) #将训练数据调整成pytorch中lstm算法的输入维度
    y_train = y_train.reshape(-1, 1, predstep)  #将目标值调整成pytorch中lstm算法的输出维度
    
    
    
     #将ndarray数据转换为张量，因为pytorch用的数据类型是张量
    
    x_train = t.from_numpy(x_train)
    y_train = t.from_numpy(y_train)
    
    rnn=[]
    '''   
    
    class BiLSTM(nn.Module):
      def __init__(self):
          super(BiLSTM, self).__init__()
          self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
          # fc
          self.fc = nn.Linear(n_hidden * 2, n_class)
  
      def forward(self, X):
          # X: [batch_size, max_len, n_class]
          batch_size = X.shape[0]
          input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]
  
          hidden_state = t.randn(1*2, batch_size, n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = t.randn(1*2, batch_size, n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
  
          outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
          outputs = outputs[-1]  # [batch_size, n_hidden * 2]
          model = self.fc(outputs)  # model : [batch_size, n_class]
          return model'''
    #二、创建LSTM模型
    class BiLSTM(nn.Module):
        def __init__(self):
            super(BiLSTM,self).__init__() #面向对象中的继承
            self.lstm = nn.LSTM(trainstep,trainstep,1, bidirectional=True) #输入数据特征维度，隐藏层维度，LSTM串联个数，第二个LSTM接收第一个的计算结果
            self.out = nn.Linear(trainstep*2,predstep) #线性拟合，接收数据的维度为6，输出数据的维度为1
        def forward(self,x):
            x1,_ = self.lstm(x)
            a,b,c = x1.shape
            out = self.out(x1.view(-1,c)) #因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
            out1 = out.view(a,b,-1) #因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
            return out1
    
    
    rnn = BiLSTM()
    #参数寻优，计算损失函数
    
    optimizer = t.optim.Adam(rnn.parameters(),lr = 0.02)
    loss_func = nn.MSELoss()
    
     #三、训练模型
    
    for i in range(100):
        var_x = Variable(x_train).type(t.FloatTensor)
        var_y = Variable(y_train).type(t.FloatTensor)
        out = rnn(var_x)
        loss = loss_func(out,var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print('Epoch:{}, Loss:{:.5f}'.format(i+1, loss.item()))
             
            
    
    #四、模型测试
    
    #准备测试数据
    #得到用于验证的数据集
    dataX, dataY = creat_dataset(uu,(q+1)*predstep,trainstep,predstep)  #第一维是x 第二维是t
    
    
    dataX1 = dataX.reshape(-1,1,trainstep)
    dataX2 = t.from_numpy(dataX1)
    var_dataX = Variable(dataX2).type(t.FloatTensor)
    
     
    
    pred = rnn(var_dataX)
    pred_test = pred.view(-1).data.numpy()  #转换成一维的ndarray数据，这是预测值dataY为真实值
    
    dataY=dataY.reshape(-1)
    
    result_pred.append(pred_test)
    result_real.append(dataY)
    
    #计算误差
    npreal=np.array(result_real)
    nppred=np.array(result_pred)
    MSE=met.mean_squared_error(result_pred,result_real)
    MAE=met.mean_absolute_error(result_pred,result_real)
    RMSE=met.mean_squared_error(result_pred,result_real)**0.5
    MAD=met.median_absolute_error(result_pred,result_real)
    R2=R2func(nppred,npreal)
    adjR2=adjusted_R2func(result_pred,result_real,trainstep)
    index1=np.argwhere(npreal==0)
    npreal=np.delete(npreal,index1)
    nppred=np.delete(nppred,index1)
    index1=np.argwhere(npreal==0)
    npreal=np.delete(npreal,index1)
    nppred=np.delete(nppred,index1)
    MAPE=np.mean(np.abs((nppred-npreal)/npreal))
    print('MSE',MSE)
    print('MAE',MAE)
    print('RMSE',RMSE)
    print('MAD',MAD)
    print('MAPE',MAPE)
    print('R2',R2)
    print('adjR2',adjR2)
    



