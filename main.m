load('three_cluster_example.mat');
x1 = -8:1:8;
%need some test samples?
test_data = [mvnrnd(mu_1,sigma_1,50);mvnrnd(mu_2,sigma_2,50);mvnrnd(mu_3,sigma_3,50)];
test_label = [ones(50,1);2*ones(50,1);3*ones(50,1)];

%maxEnt
[ERROR pmf ENT_P ParamPR]= semi_learning_toy(data,label,test_data,test_label,[1],[2 3],20,false,2);
w0_maxE = ParamPR.alpha0;
w1_maxE = ParamPR.alpha(1);
w2_maxE = ParamPR.alpha(2);
x2_maxE = -w0_maxE/w2_maxE+(-w1_maxE/w2_maxE)*x1;

%plot decision boundary
plot(data(11:end,1),data(11:end,2),'bo')
hold on
plot(data(1:10,1),data(1:10,2),'gx',data(11:20,1),data(11:20,2),'rx','linewidth',2)
x2_maxE = w0_maxE/w2_maxE+(-w1_maxE/w2_maxE)*x1;
plot(x1,x2_maxE,'--k','linewidth',2);
legend('unlabeled samples','class 1 labels','class 2 labels','maxEnt');
axis([-8 8 -8 8]);
hold off;