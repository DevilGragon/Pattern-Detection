%��ȷ��95%����
%%��ѵ�����οɴ�97%
clc;
close all;
num = input('����ѵ���������� num = '); %ѡȡѵ����������
randnum = randperm(150);    %�������num����
SamNum = 75;    %��������������
TestSamNum = 75;    %��������������
% ForcastSamNum=2;   %Ԥ������������
HiddenUnitNum=8;    %�м�����ڵ�����
InDim=4;      %���������ά��
OutDim=3;      %���������ά��
%% ѵ�������� ���ֻ���4������
data = load('iris_data.txt'); %����IRIS����
Test = data(randnum(1:num),:);
Test = Test';
Train = data(randnum(num+1:150),:);
Train = Train';

% Test=[1.24,1.80;1.28,1.84;1.40,2.04]';%%�������� 2*3
SamOut=[repmat([1;0;0],1,25) repmat([0;1;0],1,25) repmat([0;0;1],1,25)];
% SamOut=[ones(1,9),zeros(1,6);zeros(1,9),ones(1,6)];
SamIn=Train;
%��һ��
[Train,ps]=mapminmax(Train,0,1);
% Test=mapminmax('apply',Test,ps);
Test=mapminmax(Test,0,1);
%% bp������ѵ��
%������Ȩ�ؼ���ֵ
W1=rand(HiddenUnitNum,InDim);
B1=rand(HiddenUnitNum,1);
W2=rand(OutDim,HiddenUnitNum);
B2=rand(OutDim,1);
%lrΪѧϰЧ��
lr=0.01;
E0=0.0000001;     %Ŀ�����
MaxEpochs=10000;    %���ѵ������
ErrHistory=[];

for step=1:MaxEpochs
    %     HiddenOut=logsig(W1*P+repmat(B1,1,SamNum));  %���������
    %     NetworkOut=logsig(W2*HiddenOut+repmat(B2,1,SamNum));  %��������
    HiddenOut=logsig(W1*Train);  %���������
    NetworkOut=logsig(W2*HiddenOut);  %��������
    Error=SamOut-NetworkOut;
    SSE=sumsqr(Error);
    
    ErrHistory=[ErrHistory  SSE];
    
    if SSE<E0,break,end
    
    %% ����6����BP����ĺ��ĳ���
    % ������Ȩֵ����ֵ�����������������ݶ��½�ԭ��������ÿһ����̬������
    %     %���������������֮���Ȩֵ����ֵ��������
    %     %���������������֮���Ȩֵ����ֵ��������
    delta2=NetworkOut.*(1-NetworkOut).*(SamOut-NetworkOut);
    w=W2;
    W2= W2+lr*(HiddenOut*delta2')';
    B2=w;%��ֵ
    %�����������Ȩϵ��
    delta1=HiddenOut.*(1-HiddenOut).*(W2'*delta2);
    w=W1;
    W1=W1+lr*(Train*delta1')';
    B1=w;
end

HiddenOut=logsig(W1*Test); % ������������ս��
NetworkOut=logsig(W2*HiddenOut);    % �����������ս��

NetworkOut(find(NetworkOut<=0.5))=0;
NetworkOut(find(NetworkOut>=0.5))=1;
NetworkOut

Result =~sum(abs(NetworkOut-SamOut));
Percent1 = sum(Result)/length(Result)