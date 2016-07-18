clear all;
close all;
clc;

X = load('iris_data.txt'); %导入IRIS数据
iris_1 = X(1:50,:);
iris_2 = X(51:100,:);
iris_3 = X(101:150,:);
X=X(:,1:4);
[n,d]=size(X);

%显示数据
plot3(iris_1(:,1),iris_1(:,2),iris_1(:,3),'r+');
hold on;
plot3(iris_2(:,1),iris_2(:,2),iris_2(:,3),'g+');
plot3(iris_3(:,1),iris_3(:,2),iris_3(:,3),'b+');
grid on;
 
%步骤1(确定控制参数及设置代表点)
c = 1;  %初始类别数
K = 3;  %聚类期望数
Thita_n = 20;  % 一个聚类中的最少样本数
Thita_s = 0.5; %标准偏差控制参数，用于控制分裂
Thita_c = 2; %类间距离控制参数，用于控制合并
L = 0;  %每次迭代允许合并的最大聚类对数
I = 10;  %允许迭代的次数
J = 1;   %当前迭代次数
 
%初始聚类数为c及聚类中心
Z = zeros(c,d);
rand_num = randperm(n);
for i = 1:c
    Z(i,:) = X(rand_num(i),:);
end
 
%设w为n行2列的矩阵，表示第w(j,2)表示第j个样本属于哪一类
w=zeros(n,2);
for i=1:n
    w(i,1)=i;
end
 
step2 = true;
 
while 1
    while step2
        %设N为c行2列矩阵，表示每一类中样本数目
        N = zeros(c,2);
        for i = 1:c
            N(i,1) = i;
        end
 
        %步骤2:计算每个样本与聚合中心距离
        for i = 1:n
            x = X(i,:);
            for j = 1:c
                y = Z(j,:);
                %选取与c个中心中距离最小的
                dist=sqrt(sum((x-y).^2));
                if j == 1
                    mini = dist;
                    min_id = 1;
                else
                    if dist < mini
                        mini = dist;
                        min_id = j;
                    end
                end
            end
            % 将第i个样本归入第j个聚类中心
            w(i,2) = min_id;
            N(min_id,2) = N(min_id,2) + 1;
        end
        %步骤3:判断各类样本数是否满足条件
        k = c;
        step2 = false;
        for i = 1:c
            if N(i,2) < Thita_n
                k = k - 1;
                step2  = true;
            end
        end
        c = k;
    end
 
    %步骤4:计算修改聚合中心
    Z = zeros(c,d);
    for i = 1:n
        id = w(i,2);%表示第i个样本属于第几类
        Z(id,:) = Z(id,:) + X(i,:);
    end
 
    for i = 1:c
        if N(i,2)~= 0
            Z(i,:) = Z(i,:) / N(i,2);
        end
    end
 
    %步骤5:计算类内距离平均值
    D = zeros(c,2);
    for i = 1:c
        D(i,1) = i;
    end
    for i = 1:n
        id = w(i,2);
        x = X(i,:);
        y = Z(id,:);
        dist = sqrt(sum((x-y).^2));
        D(id,2) = D(id,2) + dist;
    end
 
    for i = 1:c
        if N(i,2)~= 0
            D(i,2) = D(i,2) / N(i,2);
        end
    end
 
    %步骤6:计算类内总平均距离全部样本对其相应聚类中心的总平均距离）
    D1 = 0;
    for i = 1:c
        D1 = D1 + N(i,2) * D(i,2);
    end
    D1 = D1 / n;
 
    %步骤7:判别分裂、合并及迭代运算等步骤
    step8 = false;
    step11 = false;
    if J < I
        if c <= K/2
            step8 = true;
        else
            if mod(J,2) == 0 || c >= 2*K
                step11 = true;
            else
                step8 = true;
            end
        end
    else
        Thita_c = 0;
        step11 = true;
    end%end step7
 
    %步骤8-10:分裂
    if step8
        %步骤8:计算每个聚合的标准偏差向量
        xgm=zeros(c,d);%标准偏差向量矩阵
        for i = 1:n
            id = w(i,2);
            x = X(i,:);
            y = Z(id,:);
            xgm(id,:) = xgm(id,:)+(x-y).^2;
        end
        for i = 1:c
            if N(i,2) ~= 0
                xgm(i,:) = sqrt(xgm(i,:) / N(i,2));
            end
        end
 
        %步骤9:每个聚合的最大标准偏差分量
        xgm_max = zeros(c,1);
        max_id = zeros(c,1);
        for i = 1:c
            [xgm_max(i),max_id(i)] = max(xgm(i,:));
        end
 
        %步骤10:考查xgm_max,类分裂是否满足条件
        for i = 1:c
            if xgm_max(i) > Thita_s
                if(D(i,2)>D1 && N(i,2)>2*(Thita_n+1)) || c<=K/2
                    rand_k = rand;
                    if rand_k == 0
                        rand_k = 0.95;
                    end
                    r = zeros(1,d);
                    r(max_id(i)) = rand_k * xgm_max(i);
                    ZZ = Z;
                    Z = zeros(c+1,d);%新的聚类中心
                    for j = 1:c+1
                        if j < i
                            Z(j,:) = ZZ(j,:);
                        else
                            if j == i
                                Z(j,:) = ZZ(j,:) - r;
                                Z(j+1,:) = ZZ(j,:) + r;
                            else
                                if j >= i + 2
                                    Z(j,:) = ZZ(j-1,:);
                                end
                            end
                        end
                    end
                    c = c + 1;
                    J = J + 1;
                    step2 = true;
                    step11 = false;
                else
                    step11 = true;
                end
            else
                step11 = true;
            end
        end %end step10
 
    end%end step8-10
 
    %步骤11-13:合并
    if step11
        %步骤11:计算两两聚合中心间的距离
        %步骤12-13:比较两两聚类中心距离，并把小于Thita_c的按递增次序排队
        D = zeros(c,c);
        add = true;
        i = 1;
        while i <= c
            for j = i+1:c
                x=Z(i,:);
                y=Z(j,:);
                D(i,j) = sqrt(sum((x-y).^2));
                if D(i,j) < Thita_c
                    ZZ = Z;
                    Z = zeros(c-1,d);
                    for id = 1:c-1
                        if id < min(i,j)
                            Z(id,:) = ZZ(id,:);
                        end
                        if id == min(i,j)
                            Z(id,:) = (N(i,2)*ZZ(i,:)+N(j,2)*ZZ(j,:)) /(N(i,2)+N(j,2));
                        end
                        if id > min(i,j) && id < max(i,j)
                            Z(id,:) = ZZ(id,:);
                        end
                        if id >= max(i,j)
                            Z(id,:) = ZZ(id+1,:);
                        end
                    end
                    c = c - 1;
                    add = false;
                    break;
                end
            end
            if add
                i = i + 1;
            else
                add = true;
                i = 1;
            end
        end
 
        %步骤14:判断算法是否结束
        if J < I
            J = J + 1;
            step2 = true;
        else
            break;
        end
    end%step11-13
     
end %end while

figure;
hold on;
grid on;
%输出结果
disp('分类结果：');
dsp1='';
for i=1:c
    dsp2='';
    shown=false;
    for j=1:d
        if j >= d/2 && shown == false
            shown=true;
            dsp1=['  ',num2str(i),'   '];
        else
            dsp1=[dsp1,'   '];
        end
 
        dsp2=[dsp2,'x',num2str(j),'    '];
    end
    disp(['    类别     ',dsp1]);
    disp(['  样本序号     ',dsp2]);
    for idx=1:n
        if i==w(idx,2)
            dsp3='';
            for idx2=1:d
                temp=X(idx,idx2);
                numspace=6-length(num2str(temp));
                spaces='';
                for j=1:numspace
                    spaces=[spaces,' '];
                end
                dsp3=[dsp3,num2str(X(idx,idx2)),spaces];
            end
            numspace=9-length(num2str(idx));
            spaces='';
            for j=1:numspace
                spaces=[spaces,' '];
            end
            disp(['     ',num2str(idx),spaces,dsp3]);
            if i == 1
                plot3(X(idx,1),X(idx,2),X(idx,3),'r.');
            elseif i == 2
                plot3(X(idx,1),X(idx,2),X(idx,3),'g.');
            elseif i == 3
                plot3(X(idx,1),X(idx,2),X(idx,3),'b.');
            end
        end
    end
    disp(' ');
end
 
%计算分类正确度
sum = 0;
for i = 1:n
    if i <= 50
        if w(i,2) ~= 1
            sum = sum + 1;
        end
    else
        if i >50 && i <=100
            if w(i,2) ~= 2
                sum = sum + 1;
            end
        else
            if w(i,2) ~= 3
                sum = sum + 1;
            end
        end
    end
end

for i = 1: c
    plot3(Z(i,1),Z(i,2),Z(i,3),'ko');
end
error = sum / n