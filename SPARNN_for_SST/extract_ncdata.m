function output = extract_ncdata(path)
%EXTRACT_NCDATA 此处显示有关此函数的摘要
%输入nc文件的路径，输出nc文件的一些信息并把结果读取到内存中
%https://ww2.mathworks.cn/help/matlab/network-common-data-form.html
ncdisp(path);
output.finfo = ncinfo(path);
output.time  = ncread(path,'time');
output.lat  = ncread(path,'lat');
output.lon  = ncread(path,'lon');
output.sst = ncread(path,'sst');
end

