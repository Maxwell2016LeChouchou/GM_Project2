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

%% discriminant analysis classifier
Factor = TreeBagger(500, x', y'+1);
[Predict_label,Scores] = predict(Factor, x');

%% svm
Factor = svmtrain(x', y'+1);
predict_label = svmclassify(Factor, xTest');
size(find(yTest+1-predict_label'==0),2)/400;

%% logistic regression
[weight,bias]= logtrain(x,y);
predict_label=logclassify(weight,bias,xTest);