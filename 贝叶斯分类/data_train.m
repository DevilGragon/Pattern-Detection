function [avr,var]=data_train(datas)   %ѵ������
b=size(datas,2);      %����������
f1=datas(:,1:b);      
avr=mean(f1)';        %���ֵ
var=cov(f1);          %��Э�������