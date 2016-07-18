clc
%IRIS数据前两类线性可分，后两类线性不可分
data = load('iris_data.txt'); %导入IRIS数据
iris_1 = data(1:50,:);
iris_2 = data(51:100,:);
iris_3 = data(101:150,:);

num = input('输入训练样本个数 num = '); %选取训练样本个数

%存取前两类数据
linear_data = data(1:100,:);
randnum = randperm(100); %产生100个随机数
%随机从前两类数据中抽取num个数据训练感知器
selected_linear_data = linear_data(randnum(1:num),:);
[X, Y] = size(selected_linear_data); %获得训练数据的行、列值

%设置最大迭代次数
max_iterator = 10000;

T = ones(1, Y);
s = size(T, 1);
w = zeros(s, X);
b = ones(s, 1);
e = ones(s, Y);
 
%开始迭代
for i = 1 : max_iterator
    index = mod(i, Y);
    if(index == 0)
        index = 4;
    end
    p = selected_linear_data(:,index);
    t = T(:,index);
    a = w * p + b;
    for j = 1 : s
        if a(j) >= 0
            a(j) = 1;
        else
            a(j) = 0;
        end
    end
    e(:,index) = (t - a) * 0.5;
    w = w + e(:,index) * p';
    b = b + e(:,index);
    if(e == 0)
        break;
    end
end