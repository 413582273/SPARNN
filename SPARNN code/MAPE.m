function mape= MAPE(pre,tar)
%RMSE 此处显示有关此函数的摘要
%   此处显示详细说明
%   输入预测和目标值
%     rmse.rmse=0;
%     mae.pre=pre(:);
%     mae.tar=tar(:);
    pre=reshape(pre,1,[]);
    tar=reshape(tar,1,[]);
    %找出0元素
    zeroindex=find(tar==0);
    pre(zeroindex)=[];
    tar(zeroindex)=[];
    
    N=size(pre,1);
    M=size(pre,2);
    mape=0;
    for i=1:N
        for j=1:M
            mape=mape+abs((pre(i,j)-tar(i,j))/tar(i,j))/N/M;
        end
    end
%     for i=1:N
%         rmse.rmse=rmse.rmse+((rmse.pre(i)-rmse.tar(i))^2/N)^0.5;
%     end
end

