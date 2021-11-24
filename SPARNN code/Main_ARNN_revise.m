clc;
clear;
close all;
warning('off');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%    data preparation   %%%%%%%%%%%%%%%%%%%%%%%  
%%%%%%%%%%%%%%%%%%%%%%%%    Input any time-series data   %%%%%%%%%%%%%%%%%%%%

name1='.\data\kssolution256x0-200t01';
%load Y;
name=[name1,'.mat'];
load(name,'uu');
Y=uu';
%load t;
name=[name1,'-t.mat'];
load(name,'tt');
t=tt';
%load x;
% name=[name1,'-x.mat'];
% load(name,'xx1');
% x=xx1;

group=1;
% set noise
noisestrength=0.05;
X=Y+noisestrength*2*(rand(size(Y))-0.5);% noise could be added


block=32;   %每个分块的节点数量
truncation=10; %截断的步长
predict_len=15;     % L
INPUT_trainlength=21;         %  length of training data (observed data), m > 2L
k=120;  % randomly selected variables of matrix B
%导出每个关联分块的数组索引
index = association_grouping(X,block,group);

%初始化数据收集数组
result_y=[];
result_prey=[];
result_error=[];
result_rmse=[];
result_t=[];
result_tbar=[];

%初始时间点
initial=0;
ii=2;
while ii<initial+3900                 % run each case sequentially with different initials
    tic   
    ii=ii-1;
    %初始化结果矩阵
    y_prediction=[];
    for k_block=1:size(X,2)

        Accurate_predictions=0;

    
        %把原始数据进行关联分组
        xx=X(ii:size(X,1),index(k_block,:))';
        xx_noise=xx+noisestrength*rand(size(xx));
        %记录目标变量在新分组中的索引位置
        jd_index=find(index(k_block,:),k_block);
        
        
        %traindata=xx_noise(:,1:trainlength);
        % use the most recent short term high-dimensional time-series to predict
        traindata = xx_noise(:, max(1,INPUT_trainlength-3*predict_len):INPUT_trainlength);   
        trainlength=size(traindata,2);%return the column number

        jd=k_block; % the index of target variable

        D=size(xx_noise,1);     % number of variables in the system.
        %origin_real_y=xx(jd,:);
        xx_temp=X(ii:size(X,1),:)'; 
        origin_real_y=xx_temp(jd,:);
        real_y=xx_temp(jd,max(1,INPUT_trainlength-3*predict_len):end);
        real_y_noise=real_y+noisestrength*rand(size(real_y));
        traindata_y=real_y_noise(1:trainlength);

        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%    ARNN start     %%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Given a set of fixed weights for F for each time points: A*F(X^t)=Y^t, F(X^t)=B*(Y^t)
        traindata_x_NN=NN_F2(traindata);


        w_flag=zeros(size(traindata_x_NN,1));
        A=zeros(predict_len,size(traindata_x_NN,1));   % matrix A
        B=zeros(size(traindata_x_NN,1),predict_len);   % matrix B

        predict_pred=zeros(1,predict_len-1);

        
        
        %  End of ITERATION 1:  sufficient iterations
        for iter=1:1000         % cal coeffcient B

            random_idx=sort([jd_index,randsample(setdiff(1:size(traindata_x_NN,1),jd),k-1)]);
            traindata_x=traindata_x_NN(random_idx,1:trainlength);        % random chose k variables from F(D)

            clear super_bb super_AA;
            for i=1:size(traindata_x,1)
                %  Ax=b,  1: x=pinv(A)*b,    2: x=A\b,    3: x=lsqnonneg(A,b)
                b=traindata_x(i,1:trainlength-predict_len+1)';     % 1*(m-L+1)
                clear B_w;
                for j=1:trainlength-predict_len+1
                    B_w(j,:)=traindata_y(j:j+predict_len-1);
                end
                B_para=(B_w\b)';
                B(random_idx(i),:)=(B(random_idx(i),:)+B_para+B_para*(1-w_flag(random_idx(i))))/2;
                w_flag(random_idx(i))=1;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%  tmp predict based on B  %%%%%%%%%%%%%%%%%%%%%%%%%
            clear super_bb super_AA;
            for i=1:size(traindata_x_NN,1)
                kt=0;
                clear bb;
                AA=zeros(predict_len-1,predict_len-1);
                for j=(trainlength-(predict_len-1))+1:trainlength
                    kt=kt+1;
                    bb(kt)=traindata_x_NN(i,j);
                    %col_unknown_y_num=j-(trainlength-(predict_len-1));
                    col_known_y_num=trainlength-j+1;
                    for r=1:col_known_y_num
                        bb(kt)=bb(kt)-B(i,r)*traindata_y(trainlength-col_known_y_num+r);
                    end
                    AA(kt,1:predict_len-col_known_y_num)=B(i,col_known_y_num+1:predict_len);
                end

                super_bb((predict_len-1)*(i-1)+1:(predict_len-1)*(i-1)+predict_len-1)=bb;
                super_AA((predict_len-1)*(i-1)+1:(predict_len-1)*(i-1)+predict_len-1,:)=AA;
            end

            pred_y_tmp=(super_AA\super_bb')';


            %%%%%%%%%%%%%%%%%%%%%    update the values of matrix A and Y     %%%%%%%%%%%%%%%%
            tmp_y=[real_y(1:trainlength), pred_y_tmp];
            for j=1:predict_len
                Ym(j,:)=tmp_y(j:j+trainlength-1);
            end
            BX=[B,traindata_x_NN];
            IY=[eye(predict_len),Ym];
            A=IY*pinv(BX);
            clear  union_predict_y_NN;
            for j1=1:predict_len-1
                tmp_y=zeros(predict_len-j1,1);
                kt=0;
                for j2=j1:predict_len-1
                    kt=kt+1;
                    row=j2+1;
                    col=trainlength-j2+j1;
                    tmp_y(kt)=A(row,:)*traindata_x_NN(:,col);
                end
                union_predict_y_ARNN(j1)=mean(tmp_y);
            end

            %  End of ITERATION 2: the predicting result converges.
            eof_error=sqrt(immse(union_predict_y_ARNN, predict_pred));
            if eof_error<0.0001
                break
            end

            predict_pred=union_predict_y_ARNN;
        end
        
        %收集预测数据到数组中
        y_prediction=[y_prediction;union_predict_y_ARNN(:,1:truncation)];
        
        


    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%      result display    %%%%%%%%%%%%%%%%%%%%%%     
     refx=X(INPUT_trainlength+ii+1:INPUT_trainlength+predict_len+ii-1,:)';
     result=abs(y_prediction-refx(:,1:truncation));

    
    
    result_y=[result_y,refx(:,1:truncation)];
    result_prey=[result_prey,y_prediction];
    result_error=[result_error,result(:,1:truncation)];
%     result_rmse=[result_rmse,rmse.rmse];
    result_t=[result_t,t(ii+INPUT_trainlength:ii+truncation+INPUT_trainlength-2)'];
    result_tbar=[result_tbar,t(ii+INPUT_trainlength)];
    
    
    rmse_out=RMSE(result_prey,result_y);    
    mae_out=MAE(result_prey,result_y);  
    mad_out=MAD(result_prey,result_y); 
    mape_out=MAPE(result_prey,result_y);
    R2=Rsquare(result_prey,result_y);
    adjR2=adj_Rsquare(result_prey,result_y,INPUT_trainlength);
    
    MSE1=rmse_out.rmse*rmse_out.rmse
    MAE1=mae_out.mae
    RMSE1=rmse_out.rmse
    MAD1=mad_out.mad
    MAPE1=mape_out
    R21=R2
    adj_R21=adjR2
    
    ii = ii+truncation
end


savename=['N=',num2str(size(result_error,1)),'block=',num2str(block),'prelen=', ...
    num2str(predict_len),'trainlength=',num2str(INPUT_trainlength),'k=', ...
    num2str(k),'group=',num2str(group),'noise=',num2str(noisestrength*100)];

mkdir(['.\reviseresult\',savename])

savename=['.\reviseresult\',savename,'\'];

save([savename,'result_y'],'result_y')
save([savename,'result_prey'],'result_prey')
save([savename,'result_error'],'result_error')
save([savename,'result_t'],'result_t')
save([savename,'result_tbar'],'result_tbar')
save([savename,'x'],'x')



    