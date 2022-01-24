function output = sample_ncdata(para,lon,lat,ncdata,timenow,timeend)
%   输入由extract_ncdata函数提取的数据
% 
% index_lat_min=min(find(ncdata.lat>=lat(1)-0.1))
% index_lat_max=max(find(ncdata.lat<=lat(2)+0.1))
% index_lon_min=min(find(ncdata.lon>=lon(1)-0.1))
% index_lon_max=max(find(ncdata.lon<=lon(2)+0.1))


index_lat_min=find(ncdata.lat==lat(1));
index_lat_max=find(ncdata.lat==lat(2));
index_lon_min=find(ncdata.lon==lon(1));
index_lon_max=find(ncdata.lon==lon(2));

index_time=[timenow:para.train_interval:timeend];

output.sst=ncdata.sst(index_lon_min:index_lon_max,...
    index_lat_min:index_lat_max,...
    index_time);


output.time=ncdata.time(index_time);
output.lat=ncdata.lat(index_lat_min:index_lat_max);
output.lon=ncdata.lon(index_lon_min:index_lon_max);
end

