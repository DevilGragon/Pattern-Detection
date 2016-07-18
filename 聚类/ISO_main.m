clear all;
close all;
clc;

X = load('iris_data.txt'); %����IRIS����
iris_1 = X(1:50,:);
iris_2 = X(51:100,:);
iris_3 = X(101:150,:);
X=X(:,1:4);
[n,d]=size(X);

%��ʾ����
plot3(iris_1(:,1),iris_1(:,2),iris_1(:,3),'r+');
hold on;
plot3(iris_2(:,1),iris_2(:,2),iris_2(:,3),'g+');
plot3(iris_3(:,1),iris_3(:,2),iris_3(:,3),'b+');
grid on;
 
%����1(ȷ�����Ʋ��������ô����)
c = 1;  %��ʼ�����
K = 3;  %����������
Thita_n = 20;  % һ�������е�����������
Thita_s = 0.5; %��׼ƫ����Ʋ��������ڿ��Ʒ���
Thita_c = 2; %��������Ʋ��������ڿ��ƺϲ�
L = 0;  %ÿ�ε�������ϲ������������
I = 10;  %��������Ĵ���
J = 1;   %��ǰ��������
 
%��ʼ������Ϊc����������
Z = zeros(c,d);
rand_num = randperm(n);
for i = 1:c
    Z(i,:) = X(rand_num(i),:);
end
 
%��wΪn��2�еľ��󣬱�ʾ��w(j,2)��ʾ��j������������һ��
w=zeros(n,2);
for i=1:n
    w(i,1)=i;
end
 
step2 = true;
 
while 1
    while step2
        %��NΪc��2�о��󣬱�ʾÿһ����������Ŀ
        N = zeros(c,2);
        for i = 1:c
            N(i,1) = i;
        end
 
        %����2:����ÿ��������ۺ����ľ���
        for i = 1:n
            x = X(i,:);
            for j = 1:c
                y = Z(j,:);
                %ѡȡ��c�������о�����С��
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
            % ����i�����������j����������
            w(i,2) = min_id;
            N(min_id,2) = N(min_id,2) + 1;
        end
        %����3:�жϸ����������Ƿ���������
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
 
    %����4:�����޸ľۺ�����
    Z = zeros(c,d);
    for i = 1:n
        id = w(i,2);%��ʾ��i���������ڵڼ���
        Z(id,:) = Z(id,:) + X(i,:);
    end
 
    for i = 1:c
        if N(i,2)~= 0
            Z(i,:) = Z(i,:) / N(i,2);
        end
    end
 
    %����5:�������ھ���ƽ��ֵ
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
 
    %����6:����������ƽ������ȫ������������Ӧ�������ĵ���ƽ�����룩
    D1 = 0;
    for i = 1:c
        D1 = D1 + N(i,2) * D(i,2);
    end
    D1 = D1 / n;
 
    %����7:�б���ѡ��ϲ�����������Ȳ���
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
 
    %����8-10:����
    if step8
        %����8:����ÿ���ۺϵı�׼ƫ������
        xgm=zeros(c,d);%��׼ƫ����������
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
 
        %����9:ÿ���ۺϵ�����׼ƫ�����
        xgm_max = zeros(c,1);
        max_id = zeros(c,1);
        for i = 1:c
            [xgm_max(i),max_id(i)] = max(xgm(i,:));
        end
 
        %����10:����xgm_max,������Ƿ���������
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
                    Z = zeros(c+1,d);%�µľ�������
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
 
    %����11-13:�ϲ�
    if step11
        %����11:���������ۺ����ļ�ľ���
        %����12-13:�Ƚ������������ľ��룬����С��Thita_c�İ����������Ŷ�
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
 
        %����14:�ж��㷨�Ƿ����
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
%������
disp('��������');
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
    disp(['    ���     ',dsp1]);
    disp(['  �������     ',dsp2]);
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
 
%���������ȷ��
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