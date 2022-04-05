function [mylen] = threedoor(Tim,change)

%输入Tim 为随机实验次数，change表示是否换门；mylen是猜中的次数
%____________ 例子 _____________________________________________________%
% t = cputime;
% for i = 1 : 100
%    mylen(i) =  threedoor(10000,0);
% end
% disp('Without changing the door, the Probability to win the car is:');
% flen = mean(mylen);
% disp(flen/10000);
% disp('The time used in the computation:')
% disp(cputime - t)

%___________ 输出 _____________________________________________________% 
% Without changing the door, the Probability to win the car is:
%    3.3368e+03
% 
% The time used in the computation:
%    11.2969


%% 随机初始化 Tim 次实验
% a 为 Tim 次实验里，随机确定3个门后面哪个有车
% b 为 某人在 Tim次实验里，先随机选定一扇门
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

%% 如果 change==0，表示不换门；change==1表示换门
% 当 某人选定的门 n 和有车的门 m 相同时，主持人可以从另外两扇门中选一扇打开
% m~=n时，主持人只有一个选择，因为只有那扇门他可以打开
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
