function mad_out= MAD(pre,tar)
%RMSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   ����Ԥ���Ŀ��ֵ
%     rmse.rmse=0;
    mad_out.pre=pre(:);
    mad_out.tar=tar(:);
    mad_out.pre=reshape(mad_out.pre,1,[]);
    mad_out.tar=reshape(mad_out.tar,1,[]);
    
    mad_out.mad=mad(abs(mad_out.pre(:)-mad_out.tar(:)));
%     for i=1:N
%         rmse.rmse=rmse.rmse+((rmse.pre(i)-rmse.tar(i))^2/N)^0.5;
%     end
end

