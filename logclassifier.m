clear all;
clc;
close all;
%% load data and preprocession
load('classify_d5_k3_saved1.mat');
x=[class_1,class_2];
n_one=size(class_1,2);
n_zero=size(class_2,2);
[nFeatures,nSamples]=size(x);
y=[ones(1,n_one),zeros(1,n_zero)];
weight=ones(nFeatures,1);
bias=1;
nIter=100000;
alpha=0.02;
threthold=0.05;
%% 
for i=1:nIter
%% forward propagation
h = weight'*x+bias;        
% linear transform
g= 1.0 ./ (1.0 + exp(-h)); 
% activation function
error=[y(:,1:n_one).*log(g(:,1:n_one)),(1-y(:,n_one+1:n_one+n_zero)).*log(1-g(:,n_one+1:n_one+n_zero))];
J=(-1/nSamples)*sum(error);
% loss function
if J<threthold
    break
end
%% backward propagation
deltah=g-y;
% partial derivative
weight=weight-alpha*x*deltah';
% update weight
bias=bias-alpha*sum(deltah);
% update bias
end




