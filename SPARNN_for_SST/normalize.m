function output = normalize(data,standard,inverse)
%NORMALIZE ��һ������
%   data ��������  standard ��׼  inverse=1��һ��=0���һ��
if inverse==1
    output=data-standard*ones(size(data));
else
    output=data+standard*ones(size(data));
end

end

