function [weight,bias] = logtrain(x, y)

[nFeatures,nSamples]=size(x);
weight=zeros(nFeatures,1);
bias=0;
nIter=1000;
alpha=0.001;
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
deltaw=(1/nFeatures)*x*deltah';
deltab=(1/nFeatures)*sum(deltah);
weight=weight-alpha*deltaw;
% update weight
bias=bias-alpha*deltab;
% update bias
end
end



