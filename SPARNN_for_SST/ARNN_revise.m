function outputdata = ARNN_revise(inputdata,para)
% clc;
% clear;
% close all;
warning('off');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%    data preparation   %%%%%%%%%%%%%%%%%%%%%%%  
%%%%%%%%%%%%%%%%%%%%%%%%    Input any time-series data   %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%    Dataset folder: Data, including gene expression, HK hospital admission,  %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%    tempressure, SLP, Solar, stock, traffic, typhoon, wind speed    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%    example:  Lorenz  system    %%%%%%%%%%%%%%%%%%%

Y=inputdata;% coupled lorenz system
%load Y;
noisestrength=para.noisestrength;
X=Y+noisestrength*rand(size(Y));% noise could be added



INPUT_trainlength=para.trainlength;         %  length of training data (observed data), m > 2L
selected_variables_idx=[1:size(X,2)];              % selected the most correlated variables, [1:90] can be changed by personalized methods
xx=X(1:size(X,1),selected_variables_idx)';       % after transient dynamics
noisestrength=0;   % strength of noise
xx_noise=xx+noisestrength*rand(size(xx));

predict_len=para.prelength;     % L

%traindata=xx_noise(:,1:trainlength);
% use the most recent short term high-dimensional time-series to predict
traindata = xx_noise(:, max(1,INPUT_trainlength-3*predict_len):INPUT_trainlength);   
trainlength=size(traindata,2);
k=para.k;  % randomly selected variables of matrix B

jd=1; % the index of target variable

D=size(xx_noise,1);     % number of variables in the system.
origin_real_y=xx(jd,:);
real_y=xx(jd,max(1,INPUT_trainlength-3*predict_len):end);
real_y_noise=real_y+noisestrength*rand(size(real_y));
traindata_y=real_y_noise(1:trainlength);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%    ARNN start     %%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Given a set of fixed weights for F for each time points: A*F(X^t)=Y^t, F(X^t)=B*(Y^t)
traindata_x_NN=NN_F2(traindata);

%%% Randomly given a set of weights for F each time points: A*F(X)=Y, F(X)=B*Y
% clear NN_traindata;
% for i=1:trainlength
%    traindata_x_NN(:,i)=NN_F(traindata(:,i));
% end

w_flag=zeros(size(traindata_x_NN,1));
A=zeros(predict_len,size(traindata_x_NN,1));   % matrix A
B=zeros(size(traindata_x_NN,1),predict_len);   % matrix B

predict_pred=zeros(1,predict_len-1);

%  End of ITERATION 1:  sufficient iterations
for iter=1:1000         % cal coeffcient B

    random_idx=sort([jd,randsample(setdiff(1:size(traindata_x_NN,1),jd),k-1)]);
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
refx=X(INPUT_trainlength+1:INPUT_trainlength+predict_len-1,1)';
outputdata.result_prey=predict_pred;
outputdata.result_refy=refx;
end

