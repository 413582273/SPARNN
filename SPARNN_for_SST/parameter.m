function output = parameter()
%���г���Ĳ���
%   �˴���ʾ��ϸ˵��
%��ʼʱ��
output.initial_time='2019-01-01';
%��������Ĵ���
output.test_times=80;
%ÿ������������
output.test_interval=10;


%ѵ��������
% output.lon=[220,235];
% output.lat=[0,15];
% output.lon=[117.35,121.125];
% output.lat=[37,40];
%������
output.lon=[122.375,132.125];
output.lat=[21.125,30.875];
%ѵ���ĳ���
output.trainlength=60;
%Ԥ��ĳ���
output.prelength=21;
%������ʱ����
output.train_interval=1;
%ÿ�����㴰�ڵľ�γ�ȴ�С
output.dlat=4;
output.dlon=4;


%����������ʽ,1Ϊ�����2Ϊ�ͽ�����
output.group=1;
%������������Ϊ0
output.noisestrength=0;
%ÿ�������ı�������
output.block=255;

output.k=120;

%�����ļ���ַ
output.path='..\..\..\data\SST\sst.day.mean.';



end