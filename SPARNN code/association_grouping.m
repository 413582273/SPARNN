function outputdata = association_grouping(inputdata,k,p)
%���������ڶ��������ݽ��з������,����ѵ�����ݣ��������ٸ�k
%p=1Ϊ���������p=2Ϊ�������������k��
%   �˴���ʾ��ϸ˵��
if p==1
    random_idx=[];
    for i=1:size(inputdata,2)
        random_idx(i,:)=sort([i,randsample(setdiff(1:size(inputdata,2),i),k-1)]);
    end
    outputdata = random_idx;
else
    for i=1:size(inputdata,2)
        if i<=(k/2)
            idx(i,:)=[1:k];
        elseif i>=(size(inputdata,2)-k/2)
            idx(i,:)=sort([size(inputdata,2):-1:size(inputdata,2)-k+1]);
        else
            idx(i,:)=[floor(i-k/2):1:k+floor(i-k/2)-1];
        end
    end
    outputdata = idx;
end
end

