clc
%IRIS����ǰ�������Կɷ֣����������Բ��ɷ�
data = load('iris_data.txt'); %����IRIS����
iris_1 = data(1:50,:);
iris_2 = data(51:100,:);
iris_3 = data(101:150,:);

num = input('����ѵ���������� num = '); %ѡȡѵ����������

%��ȡǰ��������
linear_data = data(1:100,:);
randnum = randperm(100); %����100�������
%�����ǰ���������г�ȡnum������ѵ����֪��
selected_linear_data = linear_data(randnum(1:num),:);
[X, Y] = size(selected_linear_data); %���ѵ�����ݵ��С���ֵ

%��������������
max_iterator = 10000;

T = ones(1, Y);
s = size(T, 1);
w = zeros(s, X);
b = ones(s, 1);
e = ones(s, Y);
 
%��ʼ����
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