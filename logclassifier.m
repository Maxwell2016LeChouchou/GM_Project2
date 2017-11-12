clear all;
clc;
close all;
%% load data and preprocession
load('classify_d5_k3_saved1.mat');
inputx=[class_1,class_2];
n_one=size(class_1,2);
n_zero=size(class_2,2);
[nFeatures,nSamples]=size(inputx);
inputy=[ones(1,n_one),zeros(1,n_zero)];
% split data into test samples and training samples
[x xTest y yTest] = splitData(inputx', inputy');
clear inputx, inputy;

%initialize variables 
weight=ones(nFeatures,1);
bias=1;
nIter=10000;
alpha=0.001;
threthold=0.05;

%% training process
for i=1:nIter
%% forward propagation
h = weight'*x+bias;        
% linear transform
g= 1.0 ./ (1.0 + exp(-h)); 
% activation function
error=[y(:,find(y==1)).*log(g(:,find(y==1))),(1-y(:,find(y==0))).*log(1-g(:,find(y==0)))];
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

%% classifier process






