clear all;
close all;
clc;

data = load('iris_data.txt'); %导入IRIS数据
iris_1 = data(1:50,:);
iris_2 = data(51:100,:);
iris_3 = data(101:150,:);

%显示数据
plot3(iris_1(:,1),iris_1(:,2),iris_1(:,3),'r+');
hold on;
plot3(iris_2(:,1),iris_2(:,2),iris_2(:,3),'g+');
plot3(iris_3(:,1),iris_3(:,2),iris_3(:,3),'b+');
grid on;

%k-means聚类
[u, re] = KMeans(data,3);  %最后产生带标号的数据，标号在所有数据的最后，意思就是数据再加一维度
[m, n] = size(re);

%最后显示聚类后的数据
figure;
hold on;
for i = 1: m
    if re(i,5) == 1   
         plot3(re(i,1),re(i,2),re(i,3),'b.');
    elseif re(i,5) == 2
         plot3(re(i,1),re(i,2),re(i,3),'r.'); 
    else 
         plot3(re(i,1),re(i,2),re(i,3),'g.');
    end
end
for i = 1: 3
    plot3(u(i,1),u(i,2),u(i,3),'kx');
end
hold off;
grid on;