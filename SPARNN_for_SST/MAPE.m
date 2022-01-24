function mape= MAPE(pre,tar)
%RMSE 此处显示有关此函数的摘要
%   此处显示详细说明
%   输入预测和目标值
%     rmse.rmse=0;
%     mae.pre=pre(:);
%     mae.tar=tar(:);
    N=size(pre,1);
    M=size(pre,2);
    mape=sum(abs(pre-tar)./tar/N*M);
%     for i=1:N
%         rmse.rmse=rmse.rmse+((rmse.pre(i)-rmse.tar(i))^2/N)^0.5;
%     end
end

