function [mylen] = threedoor(Tim,change)

%����Tim Ϊ���ʵ�������change��ʾ�Ƿ��ţ�mylen�ǲ��еĴ���
%____________ ���� _____________________________________________________%
% t = cputime;
% for i = 1 : 100
%    mylen(i) =  threedoor(10000,0);
% end
% disp('Without changing the door, the Probability to win the car is:');
% flen = mean(mylen);
% disp(flen/10000);
% disp('The time used in the computation:')
% disp(cputime - t)

%___________ ��� _____________________________________________________% 
% Without changing the door, the Probability to win the car is:
%    3.3368e+03
% 
% The time used in the computation:
%    11.2969


%% �����ʼ�� Tim ��ʵ��
% a Ϊ Tim ��ʵ������ȷ��3���ź����ĸ��г�
% b Ϊ ĳ���� Tim��ʵ��������ѡ��һ����
a = zeros(Tim,3);
b = zeros(Tim,3);
for i = 1 : Tim
    temp = unidrnd(3);
    a(i,temp) = 1;
end

for i = 1 : Tim
    temp = unidrnd(3);
    b(i,temp) = 1;
end

%% ��� change==0����ʾ�����ţ�change==1��ʾ����
% �� ĳ��ѡ������ n ���г����� m ��ͬʱ�������˿��Դ�������������ѡһ�ȴ�
% m~=nʱ��������ֻ��һ��ѡ����Ϊֻ�������������Դ�
if change ==0
    for i = 1 : Tim
        m = find(a(i,:)==1);
        n = find(b(i,:)==1);
        if m == n
            while 1
                temp = unidrnd(3);
                if temp ~= m
                    a(i,temp) = -1;
                    break;
                end
            end
        else
            a(i,6-m-n) = -1;
        end
    end
else
    for i = 1 : Tim
        m = find(a(i,:)==1);
        n = find(b(i,:)==1);
        if m == n
            while 1
                temp = unidrnd(3);
                if temp ~= m
                    a(i,temp) = -1;
                    b(i,6-temp-m) = 1;
                    b(i,m) = 0;
                    break;
                end
            end
        else
            a(i,6-m-n) = -1;
            b(i,m) = 1;
            b(i,n) = 0;
        end
    end
end
    
[vec,~] = find(a==b & a==1);
mylen = length(vec);
% if change == 0
%     disp('Without changing the door, the Probability to win the car is:');
%     disp(mylen);
% else
%     disp('Changing the door, the Probability to win the car is:');
%     disp(mylen);
% end
