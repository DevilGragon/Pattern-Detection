clear all
close all
clc
%IRIS����ǰ�������Կɷ֣����������Բ��ɷ�
data = load('iris_data.txt'); %����IRIS����
iris_1 = data(1:50,:);
iris_2 = data(51:100,:);
iris_3 = data(101:150,:);

pairs = input('����ѵ���������� pairs = ������������һ��������13��'); %ѡȡѵ���������� 

num = input('����ѵ���������� num = '); %ѡȡѵ����������

T1=zeros(num,4);
T2=zeros(num,4);
T3=zeros(num,4);
%�����ȡm��ѵ������
index=randperm(50);
for i=1:1:num       %�õ������ѵ������
         T1(i,:)=iris_1(index(i),:);
         T2(i,:)=iris_2(index(i),:);
         T3(i,:)=iris_3(index(i),:);
end

W0=zeros(1,4);
u=0.02;
e=0.01;
s=1;
k=0;
W=W0;
if pairs == 13
        while (s>=e)
            for i=1:1:num 
            k=k+1;
            if rem(k,2)==1
                err=1-W*T1(i,:)';
                W=W+u*err*T1(i,:);
            else err=0-W*T3(i,:)';   
                W=W+u*err*T3(i,:);
            end
            pf(k)=err^2;
            s=pf(k);
            end
        end
elseif pairs == 12
        while (s>=e)
            for i=1:1:num 
            k=k+1;
            if rem(k,2)==1
                err=1-W*T1(i,:)';
                W=W+u*err*T1(i,:);
            else err=0-W*T2(i,:)';   
                W=W+u*err*T2(i,:);
            end
            pf(k)=err^2;
            s=pf(k);
            end
        end
elseif pairs == 23
        while (s>=e)
            for i=1:1:num 
            k=k+1;
            if rem(k,2)==1
                err=1-W*T2(i,:)';
                W=W+u*err*T2(i,:);
            else err=0-W*T3(i,:)';   
                W=W+u*err*T3(i,:);
            end
            pf(k)=err^2;
            s=pf(k);
            end
        end
else
    disp('��������������󣬳����˳�');
    return;
end
k %ѵ������
        

r1=0;r2=0;
if pairs == 12
    for i=1:1:50
        y1(i)=W*iris_1(i,:)';
        if y1(i)>0.5
            r1=r1+1;
        end
        y2(i)=W*iris_2(i,:)';
        if y2(i)<=0.5
            r2=r2+1;
        end
    end
elseif pairs == 13
    for i=1:1:50
        y1(i)=W*iris_1(i,:)';
        if y1(i)>0.5
            r1=r1+1;
        end
        y2(i)=W*iris_3(i,:)';
        if y2(i)<=0.5
            r2=r2+1;
        end
    end
else
    for i=1:1:50
        y1(i)=W*iris_2(i,:)';
        if y1(i)>0.5
            r1=r1+1;
        end
        y2(i)=W*iris_3(i,:)';
        if y2(i)<=0.5
            r2=r2+1;
        end
    end
end
plot([1:50],y1,'r');
axis([0,50,-1.5,2.5]);
title(['num=',num2str(num)]);
hold on
plot([1:50],y2);
grid on
r1=r1/50
r2=r2/50
