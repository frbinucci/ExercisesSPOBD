clear all;
close all;

fprintf("Which method do you want to test?\n");
fprintf("1 - LASSO via ISTA\n");
fprintf("2 - LASSO via ADMM\n");
choice = input("Select an option...");
%% Part 1
% In this part of the script we provide the LASSO Solution through the ISTA
% algorithm with synthetic data.
rng(46);

N = 10; %Vector Size
M = 20; %Number of observations 

%NOTE THAT these parameters work for ADMM. 
gamma = 5e-1; %Learning rate 
alpha = 1e-2; %Regularization parameter
iterations = 1e3;

%Let's generate a sparse random signal.
A = randn(N,M);
x0=zeros(M,1);
x0([1 4 7])=1;
y = A*x0;

if choice == 1
    [x_opt,x_tracker,J,fitting_cost_matrix,regularization_cost_matrix,C] = ista(A,y,alpha,gamma,iterations,x0);
else
    [x_opt,x_tracker,J,fitting_cost_matrix,regularization_cost_matrix,C,z] = lassoAdmm(A,y,alpha,gamma,iterations,x0);
end

%Plotting the MSE and the original/estimated signal
figure
subplot(2,1,1)
semilogy(C,'linewidth',2)
ylabel('Squared error')
xlabel('Iteration index')
grid on
subplot(2,1,2)
plot(x_opt,'LineWidth',2,'Marker','o')
hold on;
plot(x0,'LineWidth',2,'Marker','x')
ylabel("Estimated Signal")
xlabel("Component N.")
grid on

%% Part 2 
% In this part of the script we evalute the sparsity of the solution for
% different values of the regularization parameter
alpha_vector = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,10];
sparsity_vector = zeros(1,numel(alpha_vector));
fitting_cost_matrix=zeros(numel(alpha_vector),iterations);
regularization_cost_matrix=zeros(numel(alpha_vector),iterations);
mse_vector = zeros(1,numel(alpha_vector));
for i=1:numel(alpha_vector)
    if choice ==1
        [x_opt,x_tracker,J,fitting_cost,regularization_cost,C] = ista(A,y,alpha_vector(i),gamma,iterations,x0);
    else
        [x_opt,x_tracker,J,fitting_cost,regularization_cost,C] = lassoAdmm(A,y,alpha_vector(i),gamma,iterations,x0);
        energy = sum(x_opt.^2);
        x_opt = wthresh(abs(x_opt),'h',1e-4*energy);
    end
    regularization_cost_matrix(i,:) = regularization_cost;
    fitting_cost_matrix(i,:) = fitting_cost;
    sparsity_vector(i) = sum(abs(x_opt)>0);
    mse_vector(i) = C(end-1);
end

figure
subplot(2,1,1);
plot(alpha_vector,sparsity_vector,'LineWidth',2,'Marker','x');
set(gca,'XScale','log');
grid on;
xlabel("Regularization Parameter");
ylabel("Number of Non-Zero Elements");
subplot(2,1,2);
semilogy(alpha_vector,mse_vector,'LineWidth',2);
set(gca,'XScale','log');
grid on;
xlabel("Regularization Parameter");
ylabel("Reconstruction MSE");

figure
for i=1:numel(alpha_vector)
    semilogy(fitting_cost_matrix(i,:),'LineWidth',2,'DisplayName',strcat('\alpha=',num2str(alpha_vector(i))));
    hold on;
    set(gca,'XScale','log');
    xlabel('Iteration Number','Fontsize',14);
    ylabel('Fitting Cost Function','Fontsize',14);
end
legend('FontSize',12)
grid on;

figure
for i=1:numel(alpha_vector)
    semilogy(regularization_cost_matrix(i,50:end),'LineWidth',2,'DisplayName',strcat('\alpha=',num2str(alpha_vector(i))));
    hold on;
    set(gca,'XScale','log');
    xlabel('Iteration Number','Fontsize',14);
    ylabel('Regularization Term','Fontsize',14);
end
legend('FontSize',12)
grid on;








