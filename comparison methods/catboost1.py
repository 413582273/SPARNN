"""
https://zhuanlan.zhihu.com/p/359143923
"""

import catboost as cat



import numpy as np 

 
import torch

import scipy.io as scio 
from sklearn import metrics as met

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#参数
trainstep=20
predstep=10
epochs = 50
noise = 0.1
# 初始化预测的数组和对比的数组
ypred=[]
yref=[]

datafile='kssolution256x0-200t01.mat'
#datafile='brusselator.mat'
#datafile='sh_data.mat'
datamat=scio.loadmat(datafile)
uu=datamat['uu'] #第一维是x 第二维是t

#uu = torch.FloatTensor(uu)
#定义一些误差评价函数
def R2func(fi_list,yi_list):
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
    xout = []
    yout = []
    shape12=np.shape(dataset)
    shape12=shape12[0]
    for j in range(shape1):
        data_x=dataset[j,i:i+look_back]
        data_y=dataset[j,i+look_back:i+look_back+predict_forward]
        xout=np.append(xout,data_x)
        yout=np.append(yout,data_y)
    xout=xout.reshape(256,-1)
    yout=yout.reshape(256,-1)
    return xout,yout #转为ndarray数据

#######################################################开始xgboost的代码

    
    
while timestart<shape2:
    

    print(timestart)

    #输出训练数据
    data_x,data_y = creat_dataset(uu_noise,timestart,trainstep,1)
    
    
    
    
    # Create an XGBoost model 
    model = cat.CatBoostRegressor(max_depth=5, learning_rate=0.1, n_estimators=100) 
     
    
    '''for data_x, data_y in train_data:
        data_x.reshape(20,-1)
    data_x.reshape(20,-1)'''
    
    model.fit(data_x, data_y) 
    
    del data_x,data_y
    
    # 循环的每一步表示向时间序列向后滑动一格
    for j in range(shape1):
        pred_input = uu_noise[j,timestart+1:timestart+trainstep+1].tolist()
        #pred_input = pred_input.reshape(-1,trainstep)
        
        for q in range(predstep):
            #输出对比的数据
            yref.append(uu[j,timestart+trainstep+1+q:timestart+trainstep+1+1+q])
            kkk=np.array(pred_input[-trainstep:],dtype=object)
            kkk=kkk.reshape(1,-1)
            seq = model.predict(kkk[-trainstep:])
            with torch.no_grad():
                pred_input.append(seq)
                #输出预测的数据
                ypred.append(seq)
                
                
                
    timestart=timestart+predstep

    #计算误差

    npreal=np.array(yref)
    nppred=np.array(ypred)
    MSE=met.mean_squared_error(ypred,yref)
    MAE=met.mean_absolute_error(ypred,yref)
    RMSE=met.mean_squared_error(ypred,yref)**0.5
    MAD=met.median_absolute_error(ypred,yref)
    R2=R2func(nppred,npreal)
    adjR2=adjusted_R2func(ypred,yref,trainstep)
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
    
    
    