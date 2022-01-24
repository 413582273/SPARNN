clc;
clear;
close all;
%��������Ԥ���Ϊ���Ľ��ж��ʱ�����ֵʵ��
%parameter�ļ��������˸��ּ�����Ҫ�Ĳ���
%SST_year�ļ����������������һ����Ҫ��Щ��ݵ����ݣ�����������
%extract_ncdata�ļ����ܹ��򿪶�Ӧ��nc�ļ�
%sample_ncdata�ļ����Ӷ�ȡ����nc�ļ�����ȡ������Ҫ�����������
%association_grouping�ļ���ARNN���������õ������ֱ����������ϻ��߾ͽ����
%ARNN�ļ���Ԥ���������
%NN_F2�ļ�����ʾ��������Ľṹ
%reshapedata,����ά����ת���ɶ�ά
%RMSE
%MAE
%MAPE



%��ȡ����
para=parameter();



%��ʼ����
% �򿪶�Ӧ��nc�ļ�
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

%��ʼ���洢���������
y=[];
yref=[];
for test_times=1:para.test_times %ʱ����ǰ����Ĵ���
rmse=[];
mae=[];
mape=[];
    %����ÿ��Ԥ��ĳ�ʼʱ��
    timenow_julian=(test_times-1)*para.test_interval+1;
%     ����ÿ��Ԥ��Ľ���ʱ��
    timeend_julian=(test_times-1)*para.test_interval+...
        (para.trainlength+para.prelength)*para.train_interval;
    
    disp([test_times,timenow_julian,timeend_julian])
    
    %����ÿ���ֲ��ļ���飬��ȡ����
    for p=1:1:floor((para.lon(2)-para.lon(1))/0.25)
        for q=1:floor((para.lat(2)-para.lat(1))/0.25)
            %���ݼ������ȡ���ݣ��Բο���Ϊ���ģ�����Ϊ0.25��
            %�������ĵ�
            centerlat=para.lat(1)+q*0.25;
            centerlon=para.lon(1)+p*0.25;
            %�������ĵ���㷶Χ
            latrange(1)=centerlat-para.dlat/2;
            latrange(2)=centerlat+para.dlat/2;
            lonrange(1)=centerlon-para.dlon/2;
            lonrange(2)=centerlon+para.dlon/2;
            
            %������ȡĿ������Ĳ���ֵ
            datacenter=sample_ncdata(para,...
                [centerlon,centerlon],[centerlat,centerlat],ncdata,...
                timenow_julian,timeend_julian);
            sstcenter=datacenter.sst;
            %�жϸõ��Ƿ�Ϊ½�ػ��ߵ��죬����ǵ������˳�����ѭ��
            if (sstcenter(1)<-200)
                disp(['�˴�Ϊ½��',num2str(centerlat),num2str(centerlon)]);
                continue
            end
            
            
            %���������ȡ�������ݣ�����ȡ������ά�����ɶ�ά����
            sstcenter_reshape=reshapedata(sstcenter);
            
            %��ȡ�ܱ�����
            data=sample_ncdata(para,lonrange,latrange,ncdata,...
                timenow_julian,timeend_julian);
            sst=data.sst;
            %���������ȡ�������ݣ�����ȡ������ά�����ɶ�ά����
            sst_reshape=reshapedata(sst);
            
            
            %�ڵ�һ�м���Ŀ������Ĳ���ֵ
            sst=[sstcenter_reshape;sst_reshape];
            
            
% %%%%%%%%%%%%%%%%%%%  ��һ������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %����ARNN����Ԥ�⣬Ϊ��������Ƚ�sst����ת��
%             sst=sst';
%             %ȥ�����е����½�ز��ֵ�����
%             length=size(sst,1);
%             sst(find(sst<-200))=[];
%             sst=reshape(sst,length,[]);
%             %��һ��
%             standard=mean(sstcenter_reshape(1:para.trainlength));
%             sst_normal=normalize(sst,standard,1);
%             
%             result=ARNN_revise(sst_normal,para);
%             
%             %���һ��
%             result.result_prey=normalize(result.result_prey,standard,0);
%             result.result_refy=normalize(result.result_refy,standard,0);
%             
%             %���Ԥ����
%             y=[y;result.result_prey];
%             % y=y';
%             yref=[yref;result.result_refy];
% %%%%%%%%%%%%%%%%%%%  ��һ������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%  ����һ������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            %����ARNN����Ԥ�⣬Ϊ��������Ƚ�sst����ת��
            sst=sst';
            %ȥ�����е����½�ز��ֵ�����
            length=size(sst,1);
            sst(find(sst<-200))=[];
            sst=reshape(sst,length,[]);
            
            result=ARNN_revise(sst,para);
            %���Ԥ����
            y=[y;result.result_prey];
            % y=y';
            yref=[yref;result.result_refy];
%%%%%%%%%%%%%%%%%%%  ����һ������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
        end
    end
    for i=1:size(yref,2)
        rmse=[rmse,RMSE(y(:,i),yref(:,i))];
        mae=[mae,MAE(y(:,i),yref(:,i))];
        mape=[mape,MAPE(y(:,i),yref(:,i))];
    end
    rmse
end





% ��ȡ��Ӧ��SST����
% data=SST_data_read(para,year)