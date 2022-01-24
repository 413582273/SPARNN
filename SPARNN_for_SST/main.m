clc;
clear;
close all;
%本程序以预测点为中心进行多个时间的数值实验
%parameter文件，定义了各种计算需要的参数
%SST_year文件，输入参数，计算一共需要哪些年份的数据，输出年份数组
%extract_ncdata文件，能够打开对应的nc文件
%sample_ncdata文件，从读取到的nc文件中提取我们需要的区域的数据
%association_grouping文件，ARNN函数里面用到，区分变量的随机组合或者就近组合
%ARNN文件，预测的主函数
%NN_F2文件，表示了神经网络的结构
%reshapedata,把三维数据转换成二维
%RMSE
%MAE
%MAPE



%读取参数
para=parameter();



%开始计算
% 打开对应的nc文件
path0=['..\..\..\..\..\data\SST\sst.day.mean.',para.initial_time(1:4),'.nc'];
ncdata0=extract_ncdata(path0);
num2str(str2num(para.initial_time(1:4))-1)
path1=['..\..\..\..\..\data\SST\sst.day.mean.',...
    num2str(str2num(para.initial_time(1:4))-1),'.nc'];
ncdata1=extract_ncdata(path1);


ncdata.sst=cat(3,ncdata1.sst,ncdata0.sst);
ncdata.time=[ncdata1.time;ncdata0.time];
ncdata.lat=ncdata0.lat;
ncdata.lon=ncdata0.lon;

clear ncdata0;
clear ncdata1;

%初始化存储结果的数组
y=[];
yref=[];
for test_times=1:para.test_times %时间往前计算的次数
rmse=[];
mae=[];
mape=[];
    %计算每次预测的初始时间
    timenow_julian=(test_times-1)*para.test_interval+1;
%     计算每次预测的结束时间
    timeend_julian=(test_times-1)*para.test_interval+...
        (para.trainlength+para.prelength)*para.train_interval;
    
    disp([test_times,timenow_julian,timeend_julian])
    
    %根据每个局部的计算块，提取数据
    for p=1:1:floor((para.lon(2)-para.lon(1))/0.25)
        for q=1:floor((para.lat(2)-para.lat(1))/0.25)
            %根据计算块提取数据，以参考点为中心，步长为0.25度
            %计算中心点
            centerlat=para.lat(1)+q*0.25;
            centerlon=para.lon(1)+p*0.25;
            %根据中心点计算范围
            latrange(1)=centerlat-para.dlat/2;
            latrange(2)=centerlat+para.dlat/2;
            lonrange(1)=centerlon-para.dlon/2;
            lonrange(2)=centerlon+para.dlon/2;
            
            %首先提取目标变量的测量值
            datacenter=sample_ncdata(para,...
                [centerlon,centerlon],[centerlat,centerlat],ncdata,...
                timenow_julian,timeend_julian);
            sstcenter=datacenter.sst;
            %判断该地是否为陆地或者岛屿，如果是岛屿则退出本次循环
            if (sstcenter(1)<-200)
                disp(['此处为陆地',num2str(centerlat),num2str(centerlon)]);
                continue
            end
            
            
            %重新组合提取到的数据，把提取到的三维数组变成二维数组
            sstcenter_reshape=reshapedata(sstcenter);
            
            %提取周边数据
            data=sample_ncdata(para,lonrange,latrange,ncdata,...
                timenow_julian,timeend_julian);
            sst=data.sst;
            %重新组合提取到的数据，把提取到的三维数组变成二维数组
            sst_reshape=reshapedata(sst);
            
            
            %在第一行加上目标变量的测量值
            sst=[sstcenter_reshape;sst_reshape];
            
            
% %%%%%%%%%%%%%%%%%%%  归一化数据  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %利用ARNN进行预测，为方便计算先将sst进行转置
%             sst=sst';
%             %去掉所有岛屿和陆地部分的数据
%             length=size(sst,1);
%             sst(find(sst<-200))=[];
%             sst=reshape(sst,length,[]);
%             %归一化
%             standard=mean(sstcenter_reshape(1:para.trainlength));
%             sst_normal=normalize(sst,standard,1);
%             
%             result=ARNN_revise(sst_normal,para);
%             
%             %逆归一化
%             result.result_prey=normalize(result.result_prey,standard,0);
%             result.result_refy=normalize(result.result_refy,standard,0);
%             
%             %输出预测结果
%             y=[y;result.result_prey];
%             % y=y';
%             yref=[yref;result.result_refy];
% %%%%%%%%%%%%%%%%%%%  归一化数据  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%  不归一化数据  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            %利用ARNN进行预测，为方便计算先将sst进行转置
            sst=sst';
            %去掉所有岛屿和陆地部分的数据
            length=size(sst,1);
            sst(find(sst<-200))=[];
            sst=reshape(sst,length,[]);
            
            result=ARNN_revise(sst,para);
            %输出预测结果
            y=[y;result.result_prey];
            % y=y';
            yref=[yref;result.result_refy];
%%%%%%%%%%%%%%%%%%%  不归一化数据  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
        end
    end
    for i=1:size(yref,2)
        rmse=[rmse,RMSE(y(:,i),yref(:,i))];
        mae=[mae,MAE(y(:,i),yref(:,i))];
        mape=[mape,MAPE(y(:,i),yref(:,i))];
    end
    rmse
end





% 读取相应的SST数据
% data=SST_data_read(para,year)