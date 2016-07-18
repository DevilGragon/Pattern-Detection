function [avr,var]=data_train(datas)   %训练函数
b=size(datas,2);      %求矩阵的列数
f1=datas(:,1:b);      
avr=mean(f1)';        %求均值
var=cov(f1);          %求协方差矩阵