clear all;
close all;

%% Ridge Regression
% In this part of the script we provide the Ridge Solution through Gradient
% Descent
rng(46);

%Setting up the problem dimensions
N = 1e2;
M = 20;

%Let's generate a random signal.
A = randn(N,M);
A = A./vecnorm(A);
x0 = randn(M,1);
noise_power = 1;

%Generation of the synthetic data through the linear observation model
y = A*x0 + sqrt(noise_power)*randn(N,1);

%Setting up the set of the regularization parameters
alpha_grid = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1,2,5,10,20,50,100];
gamma = 1e-2;
iterations = 5000;

%Tracking variables
mse = zeros(numel(alpha_grid),iterations);
reconstruction_errors = zeros(numel(alpha_grid),1);
J_ridge = zeros(numel(alpha_grid),iterations);
reconstruction_mse_ridge = zeros(numel(alpha_grid),iterations);
alpha_index=1;
for lamba=1:numel(alpha_grid)
    [opt,x,J,fitting_term,regularization_term,C,diff_centralized] = ridgeRegression(A,y,alpha_grid(alpha_index),gamma,iterations,x0);
    J_ridge(alpha_index,:) = J;
    reconstruction_mse_ridge(alpha_index,:) = C; 
    reconstruction_errors(alpha_index) = mean(((opt-x0)).^2);
    alpha_index = alpha_index+1;
end

%Plotting the disagreement with respect to the closed-form solution
figure;
semilogy(diff_centralized,'LineWidth',2,'DisplayName','Disagreement');
grid on;
l = legend();
set(l,'FontSize',14);
set(l,'interpreter','latex');
title('Disagreement w.r.t. the closed-form solution','interpreter','latex','FontSize',14);
xlabel("Iteration Index","FontSize",14,'Interpreter','latex');
ylabel("Disagreement","FontSize",14,'Interpreter','latex');

%Plotting the temporal behaviour of the cost function
figure;
for alpha_index=1:numel(alpha_grid)
    semilogy(J_ridge(alpha_index,:),'LineWidth',2,'DisplayName',strcat('$\alpha={',num2str(alpha_grid(alpha_index)),'}$'));
    hold on;
end
grid on;
l = legend();
set(l,'FontSize',14);
set(l,'interpreter','latex');
title('Cost Function vs over time (Ridge regression)','interpreter','latex','FontSize',14);
xlabel("Iteration Index","FontSize",14,'Interpreter','latex');
ylabel("J[k] Ridge","FontSize",14,'Interpreter','latex');

%Plotting the temporal behaviour of the reconstruction error
figure;
for alpha_index=1:numel(alpha_grid)
    semilogy(reconstruction_mse_ridge(alpha_index,:),'LineWidth',2,'DisplayName',strcat('$\alpha={',num2str(alpha_grid(alpha_index)),'}$'));
    hold on;
end
grid on;
xscale('log')
l = legend();
set(l,'FontSize',14);
set(l,'interpreter','latex');
title('Reconstruction MSE over time (Ridge regression)','interpreter','latex','FontSize',14);
xlabel("Iteration Index","FontSize",14,'Interpreter','latex');
ylabel("MSE Ridge [dB]","FontSize",14,'Interpreter','latex');

%Plotting the reconstruction error for different values of the
%regularization parameter
figure;
plot(alpha_grid,10*log10(reconstruction_errors),'Marker','v','LineWidth',2,'DisplayName','Reconstruction MSE');
xlabel('$\alpha$','Interpreter','latex','FontSize',14);
ylabel('MSE','Interpreter','latex','FontSize',14);
l = legend();
set(l,'interpreter','latex');
set(l,'FontSize',14);
xscale('log');
hold on;
grid on;
hold on;

%Finding the position of best ridge regression value (\alpha=1/SNR=1).
[~,best_alpha_ridge] = find(alpha_grid(alpha_grid==1));

%% Comparisons between LASSO and Ridge regression
%Regularization parameters for the LASSO problem
alpha_grid = [1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1,2,10,20,30,40,50,60];
%Simulation hyperparameters (learning rate and number of iterations)
gamma = 1e-2;
iterations = 5000;

%Tracking variables
mse_vector_lasso = zeros(numel(alpha_grid),iterations);
sparsity_vector = zeros(numel(alpha_grid),1);
J_lasso = zeros(numel(alpha_grid),iterations);

alpha_index = 1;
for alpha=alpha_grid
    [x_opt,x_tracker,J,fitting_cost_matrix,regularization_cost_matrix,C] = ista(A,y,alpha,gamma,iterations,x0);
    %semilogy(J,'LineWidth',2,'DisplayName',strcat('$\alpha={',num2str(alpha),'}$'));
    sparsity_vector(alpha_index) = sum(x_opt==0);
    grid on;
    hold on;
    J_lasso(alpha_index,:)=J;
    mse_vector_lasso(alpha_index,:) = C;
    alpha_index = alpha_index+1;
end

%Plotting the temporal behaviour reconstruction error for different values of the
%regularization parameter
figure;
for alpha_index=1:numel(alpha_grid)
    plot(mse_vector_lasso(alpha_index,:),'LineWidth',2,'DisplayName',strcat('$\alpha={',num2str(alpha_grid(alpha_index)),'}$'));
    hold on;
end
grid on;
l = legend();
set(l,'interpreter','latex');
set(l,'FontSize',14);
title('LASSO MSE over time','interpreter','latex','FontSize',14);
xlabel('Iteration Index','Fontsize',14,'interpreter','latex');
ylabel('MSE [k]','Fontsize',14,'FontSize',14,'interpreter','latex');

%Temporal behaviour of the LASSO cost function
figure;
for alpha_index=1:numel(alpha_grid)
    plot(J_lasso(alpha_index,:),'LineWidth',2,'DisplayName',strcat('$\alpha={',num2str(alpha_grid(alpha_index)),'}$'));
    hold on;
end
l = legend();
title('LASSO cost function over time','interpreter','latex','FontSize',14);
xlabel('Iteration Index','FontSize',14,'interpreter','latex');
ylabel('$J[k]$ LASSO','FontSize',14,'interpreter','latex');
grid on;
yscale('log');
set(l,'interpreter','latex');
set(l,'FontSize',14);

%Sparsity of the solutions obtained for different values of the
%regularization parameter.
figure;
plot(alpha_grid,sparsity_vector,'LineWidth',2,'Marker','o');
xscale('log')
title('LASSO solution sparsity','interpreter','latex','FontSize',14);
xlabel('Regularization Parameter $\alpha$','FontSize',14,'interpreter','latex');
ylabel('N. Components Equal to Zero','FontSize',14,'interpreter','latex');
grid on;

[~,best_alpha_lasso] = find(alpha_grid(alpha_grid==1));

%Plotting temporal behaviour of the LASSO and Ridge regression cost
%functions
figure;
%Ridge \alpha=20, index=5
semilogy(J_ridge(best_alpha_ridge,:),'LineWidth',2,'DisplayName','Ridge $\alpha=1$');
hold on;
%LASSO \alpha=20, index=1
semilogy(J_lasso(best_alpha_lasso,:),'LineWidth',2,'DisplayName','LASSO $\alpha=1$');
grid on;
l = legend();
set(l,'interpreter','latex');
set(l,'FontSize',14);
title('Cost Function of Ridge regression vs LASSO','interpreter','latex','FontSize',14);
xlabel('Iteration Index','FontSize',14,'interpreter','latex');
ylabel('J[k]','FontSize',14,'Interpreter','latex');


