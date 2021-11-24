function adjR2 = adj_Rsquare(tar,pre,p)
%https://blog.csdn.net/weixin_38100489/article/details/78175928
tar=reshape(tar,1,[]);
pre=reshape(pre,1,[]);
I=zeros(1,size(pre,2));

R2=Rsquare(tar,pre);
n=size(tar,2);
adjR2=1-(1-R2)*(n-1)/(n-p-1);
p/n



end