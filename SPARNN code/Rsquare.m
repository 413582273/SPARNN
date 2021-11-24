function R2 = rsquare(X,Ypre)
% Filname: 'rsquare.m'. This file can be used for directly calculating
% the coefficient of determination (R2) of a dataset.
%
% Two input arguments: 'X' and 'Y'
% One output argument: 'R2'
%
% X:    Vector of x-parameter
% Y:    Vector of y-paramter
% R2:   Coefficient of determination
%
% Input syntax: rsquare(X,Y)
%
% Developed by Joris Meurs BASc (2016)

X=reshape(X,1,[]);
Y=reshape(Ypre,1,[]);
I=zeros(1,size(Y,2));
% Limitations
if length(X) ~= length(Y), error('Vector should be of same length');end
if nargin < 2, error('Not enough input parameters');end
if nargin > 2, error('Too many input parameters');end



% Calculation of R2 according to the formula: SSreg/SStot
SSreg = sum((Y - X).^2);
SStot = sum((X - I*mean(X)).^2);
R2 = 1-SSreg/SStot;


end