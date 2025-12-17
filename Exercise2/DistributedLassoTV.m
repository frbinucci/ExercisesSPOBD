clear all;
close all;

%Setting-up a random seed for reproducibility
rng(42);
%Loading the (already splitted) data-set.
load('./dataset/X_test.mat','X_test');
load('./dataset/X_train.mat','X_train');
load('./dataset/y_train.mat','y_train');
load('./dataset/y_test.mat','y_test');

N = size(X_train,1);

%Distributed algorithm parameters
N_agents=5;
%Random indexes generation (splitting across samples)
p=randi([1 N_agents], N,1);

%Assignign data to each agent
for i=1:N_agents
    A{i} = X_train(p==i,:);
    y{i} = y_train(p==i,:);
end

%ADMM parameters
rho = 10;
lambda = 10;
N_iter = 1e4;

p = size(X_train,2);
x = zeros(p,N_agents,N_iter);
u = zeros(p,N_agents,N_iter);
z = zeros(p,N_iter);
disagreement_centralized = zeros(N_iter,1);
disagreement_matrix = zeros(N_iter,1);

%Retrieving the centralized solution
opt = lassoAdmm(X_train,y_train,lambda,rho,N_iter);

%Generating the network
adj = generateAdjMatrix(N_agents);
degrees = sum(adj,2);

%Consesus matrices
L = diag(degrees)-adj;
limit = 2/max(eig(L));
eps = 0.5*limit;
w = (eye(N_agents)-eps*L);

%TV formulation hyperparameters
nu = 2;
mu = eps/nu;

for k=1:N_iter-1
    for i=1:N_agents
        Ai = A{i};
        yi = y{i};
        prox_arg = 0;
        for j=1:N_agents
            prox_arg = prox_arg+w(i,j)*x(:,j,k);
        end
        prox_arg = prox_arg+(mu/k)*Ai'*(yi-Ai*x(:,i,k));
        x(:,i,k+1)=wthresh(prox_arg,'s',(lambda*(mu/k))/(N_agents));
    end

    d_sum = 0;
    for i=1:N_agents
        for j=1:N_agents
            d = sumsqr(x(:,i,k+1)-x(:,j,k+1));
            d_sum = d_sum +d;
        end
    end
    disagreement_matrix(k) = d_sum;
    disagreement_centralized(k) = sumsqr(x(:,1,k+1)-opt);
end

opt_dist = x(:,1,k);

figure
semilogy(disagreement_matrix,'LineWidth',2);
grid on;
xlabel('Iteration Index','FontSize',14);
ylabel('Disagreement Among Nodes','FontSize',14);


%Plotting the disagreement with respect to the centralized solution
figure
semilogy(disagreement_centralized,'LineWidth',2);
grid on;
xlabel('Iteration Index','FontSize',14);
ylabel('Disagreement wrt Centralized Solution','FontSize',14);

%Comparing the regression metrics of the centralized and the distributed
%solution
y_hat_centralized = X_test*opt;
y_hat_distributed = X_test*opt_dist;

fprintf("R2 metric for centralized solution = %f\n",computeR2(y_test,y_hat_centralized));
fprintf("R2 metric for distributed solution = %f\n",computeR2(y_test,y_hat_distributed));





