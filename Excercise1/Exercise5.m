clear all;
close all;

%% Target position estiation from a set of noisy measurments

%Aim of this script is to estimate the position of a target of interest
%from a set of noisy measurements acquired by N=3 antennas.

%Definition of the antennas positions
N = 3;
Px = [0,0,80];
Py = [0,50,0];
Pxy = [Px' Py'];

pos_matrix=[0,0;
    0,50;
    80,0];

%Target poistion
target_position = [20,25];

%Network adjacency matrix and laplacian
adj_matrix = [0,0,1;0,0,1;1,1,0];
L =diag(sum(adj_matrix'))-adj_matrix;
lamba_max = max(eig(L));
epsilon = 1/(lamba_max);

%Weighting matrix (for distributed gradient descent)
W = eye(N)-epsilon*L;

%Generation of the observation model
sigma =[1,1,1e-2]; %Noise power
D = sqrt(sum((pos_matrix-target_position)'.^2))+sigma.*randn(1,N);

%Distributed Gradient Descent
K = 300;
X_opt = 100*randn(N,2);
mu = 5e-2;
disagreement = zeros(K,1);
mse = zeros(K,1);
for k=1:K
    for i=1:N
        node_sum = 0;
        grad = -D(i)*((X_opt(i,:)-pos_matrix(i,:))/norm(X_opt(i,:)-pos_matrix(i,:)))+X_opt(i,:)-pos_matrix(i,:);
        for j=1:N
            node_sum = node_sum +(W(i,j)*X_opt(j,:));
        end
        X_opt(i,:) = node_sum-mu*grad;
    end
    %Tracking
    X_avg = mean(X_opt);
    disagreement(k)=(1/N)*(norm(X_opt-X_avg)^2);
    mse(k) = norm(X_opt-target_position)^2/norm(target_position)^2;
end

%% Plotting
figure
plot(mse,'LineWidth',2);
title('MSE Temporal Behaviour','FontSize',14);
grid on;
yscale('log');
ylabel('MSE Over Time','FontSize',14,'Interpreter','latex');
xlabel('Iteration Index [k]','FontSize',14,'Interpreter','latex');

figure
plot(disagreement,'LineWidth',2);
title('Disagreeent Temporal Behaviour','FontSize',14);
grid on;
yscale('log');
ylabel('Disagreement Over Time','FontSize',14,'Interpreter','latex');
xlabel('Iteration Index [k]','FontSize',14,'Interpreter','latex');

figure
gplot(adj_matrix,Pxy);
title('2D Map','FontSize',14);
hold on;
scatter(Px,Py,'g','filled');
hold on;
scatter(target_position(1),target_position(2),'r','filled');
scatter(X_avg(1),X_avg(2),'b','v');
xlabel('X distance [m]','FontSize',14,'Interpreter','latex');
ylabel('Y distance [m]','FontSize',14,'Interpreter','latex');
grid on;
