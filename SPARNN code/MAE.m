function mae= MAE(pre,tar)
%RMSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   ����Ԥ���Ŀ��ֵ
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

