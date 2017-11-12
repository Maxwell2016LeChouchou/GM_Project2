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
[x xTest y yTest] = splitData(inputx', inputy');


weight=zeros(nFeatures,1);
bias=1;
nIter=10000;
alpha=0.0001;
threthold=0.05;

%% training process
for i=1:nIter
%% forward propagation
h = weight'*x+bias;        
% linear transform
hypothesis= 1.0 ./ (1.0 + exp(-h)); 
% activation function
J=(-1/nSamples)*sum(y.*log(hypothesis)+(1-y).*log(1-hypothesis));
% loss function
if J<threthold
    break
end
%% backward propagation
deltah=hypothesis-y;
% partial derivative
deltaw=(1/nSamples)*x*deltah';
deltab=(1/nSamples)*sum(deltah);
weight=weight-alpha*deltaw;
% update weight
bias=bias-alpha*deltab;
% update bias
end

%% classifier process
hTest = weight'*xTest+bias; 
gTest= 1.0 ./ (1.0 + exp(-hTest)); 
p =double( (gTest >= 0.5));
size(find(yTest-p==0),2)/400;



