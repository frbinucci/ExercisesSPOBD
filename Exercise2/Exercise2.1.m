%Exercise 1: 

% 1 - Solve the distributed lasso problem by splitting
% across the examples
% 2 - Plot the temporal behaviour of the disagreement among the nodes
% 3 - Test the developed solution on the mpg-dataset
% 4 - Compare the distributed solution with the centralized one

clear all;
close all;

%Loading the MPG data-set. 
%See: https://www.kaggle.com/datasets/uciml/autompg-dataset for further details
[A_train,y_train,A_test,y_test] = loadDataset('./dataset/auto-mpg.csv');

N = size(A_train,1);
M = size(A_train,2);

%LASSO ADMM parameters
rho = 10;
alpha = 10;
iterations = 5000;

%Computing the centralized solution
[x_centr,xlasso,J,fitting_term,regularization_term,C] = lassoAdmm(A_train,y_train,alpha,rho,iterations);

%We consider splitting across samples among 5 agents.
N_agents = 5;
p = randi([1 N_agents],N,1);

% Data distribution among the agents
for i=1:N_agents
    A{i}=A_train(p==i,:);
    y{i}=y_train(p==i);
end

%ADMM initialization
x = randn(M,N_agents,iterations);
u = randn(M,N_agents,iterations);
z = randn(M,iterations);

disagreement_centralized = zeros(iterations,1);
disagreement_nodes = zeros(iterations,1);
fitting_ls_cost = zeros(iterations,1);

%Optimization loop
for k=1:iterations
    %Primal variable optimizations
    for i=1:N_agents
        Ai = A{i};
        yi = y{i};
        x(:,i,k+1) = pinv(Ai.'*Ai+rho*eye(M))*(Ai'*yi+rho*(z(:,k)-u(:,i,k)));
    end
    %Aggregation
    prox_mean = mean(x(:,:,k+1)+u(:,:,k),2);
    z(:,k+1) = wthresh(prox_mean,'s',alpha/(N_agents*rho));
    %Dual ascent
    for i=1:N_agents
        u(:,i,k+1)=u(:,i,k)+x(:,i,k+1)-z(:,k+1);
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
    fitting_ls_cost(k) = 1/2*sum(abs(A_train*mean_solution-y_train).^2);
end

%Plotting the disagreement among the nodes and the temporal disagreement
%wrt the centralized solution.
figure;
semilogy(disagreement_nodes,'LineWidth',2,'DisplayName','Nodes');
hold on;
semilogy(disagreement_centralized,'LineWidth',2,'DisplayName','Centralized');
grid on;
xlabel("Iteration Index",'FontSize',14);
ylabel('Disagreement','FontSize',14);
l = legend();
set(l,'interpreter','latex');
set(l,'FontSize',14);

% Plotting the temporal behaviour of the fitting term.
figure
semilogx(fitting_term,'LineWidth',2);
grid on;
xlabel('Iteration Index','interpreter','latex','FontSize',14);
ylabel('LS cost function','interpreter','latex','FontSize',14);

%Testing the generalization capabilities on the test-set 
%on the basis of the R2 score

x_distributed = mean_solution;

y_hat_centralized = A_test*x_centr;
y_hat_distributed = A_test*x_distributed;

fprintf("R2 metric for centralized solution = %f\n",computeR2(y_test,y_hat_centralized));
fprintf("R2 metric for distributed solution = %f\n",computeR2(y_test,y_hat_distributed));