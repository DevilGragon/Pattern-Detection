clc
irisdata=load('iris_data.txt');  %����irisԭʼ����
%�ֱ�洢3������
iris_w1=irisdata(1:50,:);
iris_w2=irisdata(51:100,:);
iris_w3=irisdata(101:150,:);
%ѡȡѵ�����ݣ�����ȡ��Ҳ�������ȡ
%%%%%����ѵ������%%%%%%%
NUM=40;   
datas_flower1=[];
datas_flower2=[];
datas_flower3=[];
%ÿ������˳��ȡ��ѡȡÿһ�����ݵ�ǰ40����Ϊѵ������
for i=1:NUM
    datas_flower1=[datas_flower1;iris_w1(i,:)];
    datas_flower2=[datas_flower2;iris_w2(i,:)];
    datas_flower3=[datas_flower3;iris_w3(i,:)];
end
%�ֱ��������������ľ�ֵʸ����Э������󣬼�ѵ��
[avr_w1,var_w1]=data_train(datas_flower1);
[avr_w2,var_w2]=data_train(datas_flower2);
[avr_w3,var_w3]=data_train(datas_flower3);
%ȡʣ�µ�������Ϊ������ʶ������
datas_flower1_10=[];
datas_flower2_10=[];
datas_flower3_10=[];
for i=(NUM+1):50
    datas_flower1_10=[datas_flower1_10;iris_w1(i,:)];
    datas_flower2_10=[datas_flower2_10;iris_w2(i,:)];
    datas_flower3_10=[datas_flower3_10;iris_w3(i,:)];
end
%%%%%%%����ʶ��������ã����������w1��w2������w1��w3������w2��w3��%%%%%%
datas_w=[datas_flower1_10;datas_flower2_10]; %��������ݺϲ�
avr1=avr_w1;
var1=var_w1;
avr2=avr_w2;
var2=var_w2;
pw1=0.5;  %%%%%%%�����������%%%%%%%%
pw2=0.5;
class1=1; %%%%%%%%%����ʶ%%%%%%%%%%
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
 
%����ʶ�𣬲�����������ȷ��
[result,accury1,accury2]=classify(datas_w,avr1,var1,avr2,var2,pw1,pw2,class1,class2)