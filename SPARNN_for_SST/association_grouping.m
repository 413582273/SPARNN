function outputdata = association_grouping(inputdata,k,p)
%本函数用于对输入数据进行分组关联,输入训练数据，采样多少个k
%p=1为随机采样，p=2为采样附近最近的k个
%   此处显示详细说明
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

