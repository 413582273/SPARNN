function output = parameter()
%运行程序的参数
%   此处显示详细说明
%起始时间
output.initial_time='2019-01-01';
%进行试验的次数
output.test_times=80;
%每次试验间隔几天
output.test_interval=10;


%训练的区域
% output.lon=[220,235];
% output.lat=[0,15];
% output.lon=[117.35,121.125];
% output.lat=[37,40];
%东海的
output.lon=[122.375,132.125];
output.lat=[21.125,30.875];
%训练的长度
output.trainlength=60;
%预测的长度
output.prelength=21;
%采样的时间间隔
output.train_interval=1;
%每个计算窗口的经纬度大小
output.dlat=4;
output.dlon=4;


%变量关联方式,1为随机，2为就近关联
output.group=1;
%噪音，正常设为0
output.noisestrength=0;
%每个计算块的变量数量
output.block=255;

output.k=120;

%数据文件地址
output.path='..\..\..\data\SST\sst.day.mean.';



end