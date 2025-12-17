%Exercise 2

% 1 - Implement an algorithm to solve the LASSO problem
% assuming to deal with a network
% 2- Implement an alternative procedure based on the penality
% of the Total Variation
% 3 - Plot the disagreement among the nodes and with respect to
% the centralized solution when dealing with a fitting problem.

clear all;
close all;

%Loading the MPG data-set. 
%See: https://www.kaggle.com/datasets/uciml/autompg-dataset for further details
[A_train,y_train,A_test,y_test] = loadDataset('./dataset/auto-mpg.csv');

N = size(A_train,1);
M = size(A_train,2);

%ADMM 
rho = 10;
alpha = 10;
iterations = 1000;

%Computing the centralized solution
[x_centr,xlasso,J,fitting_term,regularization_term,C] = lassoAdmm(A_train,y_train,alpha,rho,iterations);


%We consider splitting across samples among 5 agents.
N_agents = 5;
p = randi([1 N_agents],N,1);

%Let us define the adjacency matrix
adj_matrix = zeros(N_agents,N_agents);
adj_matrix(1,3)=1;
adj_matrix(3,1)=1;
adj_matrix(2,3)=1;
adj_matrix(3,2)=1;
adj_matrix(3,5)=1;
adj_matrix(5,3)=1;
adj_matrix(3,4)=1;
adj_matrix(4,3)=1;
adj_matrix(5,4)=1;
adj_matrix(4,5)=1;

%Graph plotting (only for visualization purposes).
x_data=[0,0.1000,0.1000,0.2000,0.2000]
y_data=[0,0.1000,0,0.1000,0];
g = graph(adj_matrix);
n_labels = {'A','B','C','D','E'};
plot(g,'XData',x_data,'YData',y_data,'NodeLabel',n_labels,'LineWidth',2);
title('Generated Network','FontSize',14,'interpreter','latex');

%Generating consensus matrix
degrees = sum(adj_matrix,2);
L = diag(degrees)-adj_matrix;
epsilon = 0.5*2/max(eig(L));
W = eye(N_agents)-epsilon*L;
N_cons_iter = 100;


% Data distribution among the agents
for i=1:N_agents
    A{i}=A_train(p==i,:);
    y{i}=y_train(p==i);
end

%Initialization
x = zeros(M,N_agents,iterations);
u = zeros(M,N_agents,iterations);
z = zeros(M,N_agents,iterations);

disagreement_centralized = zeros(iterations,1);
disagreement_nodes = zeros(iterations,1);
fitting_ls_cost = zeros(iterations,1);

%Optimization loop
for k=1:iterations
    %Primal variable optimizations
    for i=1:N_agents
        Ai = A{i};
        yi = y{i};
        x(:,i,k+1) = pinv(Ai.'*Ai+rho*eye(M))*(Ai'*yi+rho*(z(:,i,k)-u(:,i,k)));
    end
    %Aggregation
    prox_mean = W*(x(:,:,k+1).'+u(:,:,k).');
    for j=1:N_cons_iter
        prox_mean = W*prox_mean;
    end
    
    z(:,:,k+1) = wthresh(prox_mean.','s',alpha/(N_agents*rho));
    %Dual ascent
    for i=1:N_agents
        u(:,i,k+1)=u(:,i,k)+x(:,i,k+1)-z(:,i,k+1);
    end
    %Computing the disagreement among the nodes
    dis = 0;
    for i=1:N_agents
        for j=1:N_agents
            dis = dis+sum((x(:,i,k)-x(:,j,k)).^2);
        end
    end
    disagreement_nodes(k) = dis;
    disagreement_centralized(k) = mean((mean(x(:,:,k),2)-x_centr).^2);
    mean_solution = mean(x(:,:,k),2);
    fitting_ls_cost(k) = 1/2*norm(A_train*mean_solution-y_train)^2;
end

%Plotting the disagreements for the solution based on the avg consensus
%algorithm.
figure;
semilogy(disagreement_nodes,'LineWidth',2,'DisplayName','Nodes');
hold on;
semilogy(disagreement_centralized,'LineWidth',2,'DisplayName','Centralized');
grid on;
xlabel("Iteration Index",'FontSize',14,'interpreter','latex');
ylabel('Disagreement','FontSize',14,'interpreter','latex');
l = legend();
title("Consesus Solution",'interpreter','latex','FontSize',14);
set(l,'interpreter','latex');
set(l,'FontSize',14);

%% Solution based on the minimization of the TV^2.
epsilon = 0.5*2/max(eig(L));
mu = 1e-1;

disagreement_nodes_tv = zeros(iterations,1);
disagreement_centralized_tv = zeros(iterations,1);

%Optimization loop
for k=1:iterations
    %Primal variable optimizations
    for i=1:N_agents
        Ai = A{i};
        yi = y{i};
        prox_arg = 0;
        for j=1:N_agents
            prox_arg = prox_arg+W(i,j)*(x(:,j,k));
        end
        prox_arg = prox_arg+mu*Ai'*(yi-Ai*x(:,i,k));
        x(:,:,k+1) = wthresh(prox_mean.','s',(alpha*mu)/N_agents);
    end
    %Computing the disagreement among the nodes
    dis = 0;
    for i=1:N_agents
        for j=1:N_agents
            dis = dis+sum((x(:,i,k)-x(:,j,k)).^2);
        end
    end
    disagreement_nodes_tv(k) = dis;
    disagreement_centralized_tv(k) = mean((mean(x(:,:,k),2)-x_centr).^2);
    mean_solution = mean(x(:,:,k),2);
end

%Plotting the disagreements for the solution based on TV minimization
figure
semilogy(disagreement_centralized_tv,'LineWidth',2,'DisplayName','Centralized');
hold on;
semilogy(disagreement_nodes_tv,'LineWidth',2,'DisplayName','Nodes');
grid on;
xlabel('Iteration Index','Fontsize',14,'interpreter','latex');
ylabel('Disagreement','FontSize',14,'interpreter','latex');
l = legend();
set(l,'interpreter','latex');
set(l,'FontSize',14);
title("TV minimization Solution",'interpreter','latex','FontSize',14);



