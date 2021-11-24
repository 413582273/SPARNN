function mae= MAE(pre,tar)
%RMSE 此处显示有关此函数的摘要
%   此处显示详细说明
%   输入预测和目标值
%     rmse.rmse=0;
    mae.pre=pre(:);
    mae.tar=tar(:);
    N=size(mae.pre,1);
    M=size(mae.pre,2);
    mae.mae=sum(abs(mae.pre-mae.tar)/N*M);
%     for i=1:N
%         rmse.rmse=rmse.rmse+((rmse.pre(i)-rmse.tar(i))^2/N)^0.5;
%     end
end

