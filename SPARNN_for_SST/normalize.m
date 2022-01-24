function output = normalize(data,standard,inverse)
%NORMALIZE 归一化数据
%   data 输入数据  standard 基准  inverse=1归一化=0逆归一化
if inverse==1
    output=data-standard*ones(size(data));
else
    output=data+standard*ones(size(data));
end

end

