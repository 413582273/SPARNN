function rmse= RMSE(pre,tar)
%RMSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   ����Ԥ���Ŀ��ֵ
%     rmse.rmse=0;
    rmse.pre=pre(:);
    rmse.tar=tar(:);
    N=size(rmse.pre,1);
    M=size(rmse.pre,2);
    rmse.rmse=sum((rmse.pre-rmse.tar).^2/N*M)^0.5;
%     for i=1:N
%         rmse.rmse=rmse.rmse+((rmse.pre(i)-rmse.tar(i))^2/N)^0.5;
%     end
end

