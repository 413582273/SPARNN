function mape= MAPE(pre,tar)
%RMSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   ����Ԥ���Ŀ��ֵ
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

