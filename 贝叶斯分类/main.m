clc
irisdata=load('iris_data.txt');  %导入iris原始数据
%分别存储3类数据
iris_w1=irisdata(1:50,:);
iris_w2=irisdata(51:100,:);
iris_w3=irisdata(101:150,:);
%选取训练数据，数据取，也可以随机取
%%%%%设置训练个数%%%%%%%
NUM=40;   
datas_flower1=[];
datas_flower2=[];
datas_flower3=[];
%每类数据顺序取，选取每一类数据的前40个作为训练数据
for i=1:NUM
    datas_flower1=[datas_flower1;iris_w1(i,:)];
    datas_flower2=[datas_flower2;iris_w2(i,:)];
    datas_flower3=[datas_flower3;iris_w3(i,:)];
end
%分别计算三类分类器的均值矢量和协方差矩阵，即训练
[avr_w1,var_w1]=data_train(datas_flower1);
[avr_w2,var_w2]=data_train(datas_flower2);
[avr_w3,var_w3]=data_train(datas_flower3);
%取剩下的数据作为待分类识别数据
datas_flower1_10=[];
datas_flower2_10=[];
datas_flower3_10=[];
for i=(NUM+1):50
    datas_flower1_10=[datas_flower1_10;iris_w1(i,:)];
    datas_flower2_10=[datas_flower2_10;iris_w2(i,:)];
    datas_flower3_10=[datas_flower3_10;iris_w3(i,:)];
end
%%%%%%%两两识别参数设置，三种情况【w1，w2】、【w1，w3】、【w2，w3】%%%%%%
datas_w=[datas_flower1_10;datas_flower2_10]; %待检测数据合并
avr1=avr_w1;
var1=var_w1;
avr2=avr_w2;
var2=var_w2;
pw1=0.5;  %%%%%%%设置先验概率%%%%%%%%
pw2=0.5;
class1=1; %%%%%%%%%类别标识%%%%%%%%%%
class2=2;
 
% datas_w=[datas_flower1_10;datas_flower3_10];
% avr1=avr_w1;
% var1=var_w1;
% avr2=avr_w3;
% var2=var_w3;
% pw1=0.1;
% pw2=0.9;
% class1=1;
% class2=3;
 
% datas_w=[datas_flower2_10;datas_flower3_10];
% avr1=avr_w2;
% var1=var_w2;
% avr2=avr_w3;
% var2=var_w3;
% pw1=0.5;
% pw2=0.5;
% class1=2;
% class2=3;
 
%分类识别，并输出结果及正确率
[result,accury1,accury2]=classify(datas_w,avr1,var1,avr2,var2,pw1,pw2,class1,class2)