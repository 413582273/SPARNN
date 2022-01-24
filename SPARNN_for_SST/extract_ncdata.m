function output = extract_ncdata(path)
%EXTRACT_NCDATA �˴���ʾ�йش˺�����ժҪ
%����nc�ļ���·�������nc�ļ���һЩ��Ϣ���ѽ����ȡ���ڴ���
%https://ww2.mathworks.cn/help/matlab/network-common-data-form.html
ncdisp(path);
output.finfo = ncinfo(path);
output.time  = ncread(path,'time');
output.lat  = ncread(path,'lat');
output.lon  = ncread(path,'lon');
output.sst = ncread(path,'sst');
end

