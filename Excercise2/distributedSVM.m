clear all;
close all;

load cancer_dataset;

targets = cancerTargets(1,:);
targets(targets==0)=-1;
x=cancerInputs;
x = x';

N=size(x,1); 
p=size(x,2); 

lambda = 3e-2;
rho = 1e0;

%train test split
N_train = floor(N*0.7);
N_test = N - N_train;

A_train = x(1:floor(N*0.7),:);
data_test = x(floor(N*0.7)+1:N,:);

y_train = targets(1:floor(N*0.7));
y_test = targets(floor(N*0.7)+1:N);

A_train = [A_train,ones(N_train,1)];
A_train = (y_train').*A_train;

cvx_begin
variable x(p+1)
minimize sum(pos(1-A_train*x))+lambda*sum_square(x)
cvx_end

x_opt = x;

%Distributed solution with average consensus
rng(42);
%Distributed algorithm parameters
N_agents=5; 
%Random indexes generation (splitting across samples)
partition=randi([1 N_agents], N_train,1);

%Assignign data to each agent
for i=1:N_agents
    A{i} = A_train(partition==i,:);
    y{i} = y_train(partition==i);
end

%Setting up the iterative algorithms
N_iter = 100;
N_consensus_iter = 100;

%Tracking variables
x_matrix = zeros(p+1,N_agents,N_iter);
z = zeros(p+1,N_iter);
u = zeros(p+1,N_agents,N_iter);

disagreement_matrix = zeros(N_iter,1);
disagreement_centralized = zeros(N_iter,1);

%Network definition
adj = generateAdjMatrix(N_agents);
degrees = sum(adj,2);

L = diag(degrees)-adj;
limit = 2/max(eig(L));
eps = 0.5*limit;

w = (eye(N_agents)-eps*L);

for k=1:N_iter-1
    k
    for i=1:N_agents
        cvx_begin quiet
            variable x(p+1)
            minimize sum(pos(1-A{i}*x))+(rho/2)*(sum_square(x-z(:,k)+u(:,i,k)))
        cvx_end
        x_matrix(:,i,k+1)=x;
    end
    %Consensus
    prox_arg = w*(x_matrix(:,:,k+1)'+u(:,:,k)');
    for n_cons=1:N_consensus_iter
        prox_arg = w*prox_arg;
    end
    z(:,k+1) = ((N_agents*rho)/(2*lambda+N_agents*rho))*prox_arg(1,:)';
    for i=1:N_agents
        u(:,i,k+1)=u(:,i,k)+x_matrix(:,i,k+1)-z(:,k+1);
    end
    
    d_sum = 0;
    for i=1:N_agents
        for j=1:N_agents
            d = sum_square(x_matrix(:,i,k+1)-x_matrix(:,j,k+1));
            d_sum = d_sum +d;
        end
    end
    disagreement_matrix(k) = d_sum;
    disagreement_centralized(k) = sum_square(z(:,k+1)-x_opt);
end

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


%Comparing the two obtained solutions

%Centralized solution
theta_opt = x_opt(1:p);
intercept = x_opt(p+1);
y_hat_centralized = sign((data_test*theta_opt+intercept)');
accuracy_centralized = length(find(y_test==y_hat_centralized))/N_test;

%Distributed solution
theta_opt_distributed = z(1:p,k);
intercept_distributed = z(p+1,k);
y_hat_distributed = sign((data_test*theta_opt_distributed+intercept_distributed)');
accuracy_distributed = length(find(y_test==y_hat_distributed))/N_test;

fprintf("The accuracy of the centralized model is=%f\n",accuracy_centralized);
fprintf("The accuracy of the distributed model is=%f",accuracy_distributed);