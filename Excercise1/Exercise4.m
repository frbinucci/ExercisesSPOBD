clear all;
close all;
set(0, 'defaultTextInterpreter', 'latex');
% In this part we test the ISTA algorithm to perform a simple linear
% regression task. The considered dataset is the Auto-mpg data-set,
% downlodable from https://www.kaggle.com/datasets/uciml/autompg-dataset

%It represents a regression data-set where we want to predict the cars'
%consumption from the following features:

% - no. cylinders
% - displacement
% - horsepower
% - weight
% - acceleration
% - model year
% - origin
% - car name


%ISTA algorithm parameters
alpha_vector=[1e-1,5e-1,1,1e1,5e1,1e2];
gamma=5e-4;
iterations=1000;

sparsity_vector = zeros(1,numel(alpha_vector));
r2_array = zeros(1,numel(alpha_vector));
fitting_cost_matrix=zeros(numel(alpha_vector),iterations);
regularization_cost_matrix=zeros(numel(alpha_vector),iterations);

%ISTA algorithm execution for different values of the regularization
%parameter.
for i=1:numel(alpha_vector)
    [X_train,y_train,X_test,y_test] = loadDataset('./dataset/auto-mpg.csv');
    [theta_opt,theta_tracker,J,fitting_cost,regularization_cost] = ista(X_train,y_train,alpha_vector(i),gamma,iterations);
    fitting_cost_matrix(i,:)=fitting_cost;
    regularization_cost_matrix(i,:)=regularization_cost;
    sparsity_vector(i) = sum(abs(theta_opt)>0);
    X_train = X_train(:,find(theta_opt));
    X_test = X_test(:,find(theta_opt));
    theta_opt = theta_opt(find(theta_opt));
    y_hat_train = X_train*theta_opt;
    y_hat_test = X_test*theta_opt;
    r2_score = computeR2(y_test,y_hat_test);
    r2_array(i) = r2_score;
    fprintf("The R2 for alpha=%f is = %f\n",alpha_vector(i),r2_score);
end

%Plotting sparsity and R2 score for different values of alpha
figure
subplot(2,1,1)
plot(alpha_vector,sparsity_vector,'Marker','o','LineWidth',2);
set(gca,'XScale','log');
ylabel("Sparsity",'FontSize',14);
xlabel("$\alpha$",'FontSize',14);
grid on;
subplot(2,1,2);
plot(alpha_vector,r2_array,'Marker','o','LineWidth',2);
set(gca,'XScale','log');
ylabel("R2 Score",'FontSize',14);
xlabel("$\alpha$",'FontSize',14);
grid on;

%Plotting the terms of the cost function
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
    semilogy(regularization_cost_matrix(i,:),'LineWidth',2,'DisplayName',strcat('\alpha=',num2str(alpha_vector(i))));
    hold on;
    set(gca,'XScale','log');
    xlabel('Iteration Number','Fontsize',14);
    ylabel('Regularization Term','Fontsize',14);
end
legend('FontSize',12)
grid on;
