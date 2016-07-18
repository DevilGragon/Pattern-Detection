%正确率95%以上
%%多训练几次可达97%
clc;
close all;
num = input('输入训练样本个数 num = '); %选取训练样本个数
randnum = randperm(150);    %随机产生num个数
SamNum = 75;    %输入样本的数量
TestSamNum = 75;    %测试样本的数量
% ForcastSamNum=2;   %预测样本的数量
HiddenUnitNum=8;    %中间层隐节点数量
InDim=4;      %网络输入的维度
OutDim=3;      %网络输出的维度
%% 训练的样本 三种花的4个参数
data = load('iris_data.txt'); %导入IRIS数据
Test = data(randnum(1:num),:);
Test = Test';
Train = data(randnum(num+1:150),:);
Train = Train';

% Test=[1.24,1.80;1.28,1.84;1.40,2.04]';%%测试数据 2*3
SamOut=[repmat([1;0;0],1,25) repmat([0;1;0],1,25) repmat([0;0;1],1,25)];
% SamOut=[ones(1,9),zeros(1,6);zeros(1,9),ones(1,6)];
SamIn=Train;
%归一化
[Train,ps]=mapminmax(Train,0,1);
% Test=mapminmax('apply',Test,ps);
Test=mapminmax(Test,0,1);
%% bp神经网络训练
%先设置权重及阈值
W1=rand(HiddenUnitNum,InDim);
B1=rand(HiddenUnitNum,1);
W2=rand(OutDim,HiddenUnitNum);
B2=rand(OutDim,1);
%lr为学习效率
lr=0.01;
E0=0.0000001;     %目标误差
MaxEpochs=10000;    %最多训练次数
ErrHistory=[];

for step=1:MaxEpochs
    %     HiddenOut=logsig(W1*P+repmat(B1,1,SamNum));  %隐含层输出
    %     NetworkOut=logsig(W2*HiddenOut+repmat(B2,1,SamNum));  %输出层输出
    HiddenOut=logsig(W1*Train);  %隐含层输出
    NetworkOut=logsig(W2*HiddenOut);  %输出层输出
    Error=SamOut-NetworkOut;
    SSE=sumsqr(Error);
    
    ErrHistory=[ErrHistory  SSE];
    
    if SSE<E0,break,end
    
    %% 以下6行是BP网络的核心程序
    % 他们是权值（阈值）依据能量函数负梯度下降原理所作的每一步动态调整量
    %     %对输出层与隐含层之间的权值和阈值进行修正
    %     %对输入层与隐含层之间的权值和阈值进行修正
    delta2=NetworkOut.*(1-NetworkOut).*(SamOut-NetworkOut);
    w=W2;
    W2= W2+lr*(HiddenOut*delta2')';
    B2=w;%阈值
    %调整隐含层加权系数
    delta1=HiddenOut.*(1-HiddenOut).*(W2'*delta2);
    w=W1;
    W1=W1+lr*(Train*delta1')';
    B1=w;
end

HiddenOut=logsig(W1*Test); % 隐含层输出最终结果
NetworkOut=logsig(W2*HiddenOut);    % 输出层输出最终结果

NetworkOut(find(NetworkOut<=0.5))=0;
NetworkOut(find(NetworkOut>=0.5))=1;
NetworkOut

Result =~sum(abs(NetworkOut-SamOut));
Percent1 = sum(Result)/length(Result)