% clear all;
close all;
clc;
data = load('iris_data.txt');
iris_1 = data(1:50,:);
iris_2 = data(51:100,:);
iris_3 = data(101:150,:);
m=input('训练样本数m = ');       %输入训练样本数
pairs = input('输入训练样本类型 pairs = （举例：类型一三，输入13）'); %选取训练样本类型 

T1=zeros(m,4);
T2=zeros(m,4);
T3=zeros(m,4);
%随机抽取m个训练样本
index=randperm(50);
 for i=1:1:m       %得到随机的训练样本
         x1(i,:)=iris_1(index(i),:);
         x2(i,:)=iris_2(index(i),:);
         x3(i,:)=iris_3(index(i),:);
 end

p=0;r11=0;r22=0;
while p<50
a=0.5;
pf=1;
w1=rand(4,5);
w2=rand(6,1);
k=1;tt=1;
while pf>0.000001
    if k>m
        k=1;
    end
    if pairs == 12
        if mod(k,2)==1
            g1=x1(k,:);
            d=0.1;
        end
        if mod(k,2)==0
            g1=x2(k,:);
            d=0.9;
        end
    elseif pairs == 13
        if mod(k,2)==1
            g1=x1(k,:);
            d=0.1;
        end
        if mod(k,2)==0
            g1=x3(k,:);
            d=0.9;
        end
    elseif pairs == 23
        if mod(k,2)==1
            g1=x2(k,:);
            d=0.1;
        end
        if mod(k,2)==0
            g1=x3(k,:);
            d=0.9;
        end
    else
        disp('样本输入类型错误，程序退出');
        return;
    end

    g2=g1*w1;
    o1=1./(1+exp(-g2));
    o1_y=[o1 -1];
    g3=o1_y*w2;
    o2=1./(1+exp(-g3));
    % 输出层的权系数调整
    det2=a*2*(d-o2)*o2*(1-o2)*o1_y;
    w2=w2+det2';
    s=2*(d-o2)*o2*(1-o2);
    %第一层的权系数调整
    w22=w2';
    det1=a*g1'*(s*w22(:,1:5).*o1.*(1-o1));
    w1=w1+det1;
    err=d-o2;
    pf=err^2;
    k=k+1;  
    tt=tt+1;  %训练次数
end

r1=0;r2=0;
for j=1:1:50
    if pairs == 12
        t1=iris_1(j,:)*w1;
    elseif pairs == 13
        t1=iris_1(j,:)*w1;
    else
        t1=iris_2(j,:)*w1;
    end
    y1=1./(1+exp(-t1));
    y1_y=[y1 -1];
    I2=y1_y*w2;
    y2(j)=1./(1+exp(-I2));
    if y2(j)<0.5
        r1=r1+1;
    end
end
lv1=r1/50
if r1==50
    r11=r11+1;
end
for j=1:1:50
    if pairs == 12
        t1=iris_2(j,:)*w1;
    elseif pairs == 13
        t1=iris_3(j,:)*w1;
    else
        t1=iris_3(j,:)*w1;
    end
    y1=1./(1+exp(-t1));
    y1_y=[y1 -1];
    I2=y1_y*w2;
    y22(j)=1./(1+exp(-I2));
    if y22(j)>0.5
        r2=r2+1;
    end
end
lv2=r2/50
if r2==50
    r22=r22+1;
end
 p=p+1
end
r11/50
r22/50
plot([1:50],y2,'r');
axis([0,50,0,1]);
title(['m=',num2str(m)]);
hold on
plot([1:50],y22);
grid on
lv=(r11+r22)/100
