clear all;
close all;
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
N_iter = 500;

p = size(X_train,2);
x = zeros(p,N_agents,N_iter);
u = zeros(p,N_agents,N_iter);
z = zeros(p,N_iter);
disagreement_matrix = zeros(N_iter,1);
disagreement_centralized = zeros(N_iter,1);

%Retrieving the centralized solution
opt = lassoAdmm(X_train,y_train,lambda,rho,N_iter);

for k=1:N_iter-1
    for i=1:N_agents
        Ai = A{i};
        yi = y{i};
        x(:,i,k+1) = pinv((Ai'*Ai)+(rho)*eye(p))*(Ai'*yi+(rho)*(z(:,k)-u(:,i,k)));
    end
    %Consensus
    prox_arg = mean(x(:,:,k+1)+u(:,:,k),2);
    z(:,k+1) = wthresh(prox_arg,'s',lambda/(N_agents*rho));
    for i=1:N_agents
        u(:,i,k+1)=u(:,i,k)+x(:,i,k+1)-z(:,k+1);
    end

    %Disagreement calculation
    d_sum = 0;
    for i=1:N_agents
        d = sum_square(x(:,i,k+1)-z(:,k+1));
        d_sum = d_sum +d;
    end
    disagreement_matrix(k) = d_sum;
    disagreement_centralized(k) = sum_square(z(:,k+1)-opt);
end

opt_dist = z(:,k);

%Plotting the disagreement between nodes vs iteration index
semilogy(disagreement_matrix,'LineWidth',2);
grid on;
xlabel('Iteration Index','FontSize',14);
ylabel('Disagreement','FontSize',14);

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





