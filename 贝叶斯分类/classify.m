function [result,accury1,accury2]...
    =classify(datas_w,avr1,var1,avr2,var2,pw1,pw2,class1,class2)
[M,N]=size(datas_w);
result=zeros(6,M);   %分类识别结果
num1=0;
num2=0;
for i=1:M
    x=datas_w(i,1:N)';
    %分别代入2个后验概率函数中%
    h1=-0.5*(x-avr1)'*inv(var1)*(x-avr1)-0.5*log(det(var1))+log(pw1);
    h2=-0.5*(x-avr2)'*inv(var2)*(x-avr2)-0.5*log(det(var2))+log(pw2);
    %比较2个数据的大小，并判断属于哪一类%
    if((h1>h2))
        result(1,i)=x(1,1);
        result(2,i)=x(2,1);
        result(3,i)=x(3,1);
        result(4,i)=x(4,1);
        result(5,i)=class1;
    elseif(h1<h2)
        result(1,i)=x(1,1);
        result(2,i)=x(2,1);
        result(3,i)=x(3,1);
        result(4,i)=x(4,1);
        result(5,i)=class2;
    else
        result(1,i)=x(1,1);
        result(2,i)=x(2,1);
        result(3,i)=x(3,1);
        result(4,i)=x(4,1);
        result(5,i)=0;
    end   
    %对分类器决策的结果进行验证并输出结果,1表示正确，0表示错误%
    if (i<=M/2 && result(5,i)==class1)
        result(6,i)=1;
        num1=num1+1;
    elseif(i>M/2 && i<=M && result(5,i)==class2)
        result(6,i)=1;
        num2=num2+1;
    else
        result(6,i)=0;
    end
    %计算分类器识别正确率
    accury1=2*num1/M;
    accury2=2*num2/M;   
end