clear all;
close all;

rng(46);
%Let's generate a multivariate normal variable
N_train = 1e4;
N_test = 1e3;

nx = 200;
ny = 50;

sigma = randn(nx+ny);
sigma = (sigma*sigma'+eye(nx+ny));

%Generating the training-set
data_train = mvnrnd(zeros(1,nx+ny),sigma,N_train);
x_train = data_train(:,1:nx);
y_train = data_train(:,nx+1:end);

data_test = mvnrnd(zeros(1,nx+ny),sigma,N_test);

x_test = data_test(:,1:nx);
y_test = data_test(:,nx+1:end);


n_components_array = [1,2,3,4,5,6,7,8,9,10,20,30,40,50];
nmse_bn = zeros(numel(n_components_array),1);
nmse_pca = zeros(numel(n_components_array),1);
mi_bn = zeros(numel(n_components_array),1);
mi_pca = zeros(numel(n_components_array),1);
mi_x_t_bn = zeros(numel(n_components_array),1);
mi_x_t_pca = zeros(numel(n_components_array),1);
bound_bn = zeros(numel(n_components_array),1);
bound_pca = zeros(numel(n_components_array),1);
mse_bn = zeros(numel(n_components_array),1);
mse_pca = zeros(numel(n_components_array),1);

index = 1;

for nz=n_components_array
    A_pca = pca(x_train);
    A_pca = A_pca(:,1:nz);
    z_pca = x_train*A_pca;
    z_pca_test = x_test*A_pca;
    
    theta_ls = lmmse("fit","x",z_pca,"y",y_train);
    y_hat = lmmse("predict","x",z_pca_test,"theta",theta_ls);
    
    [nmse_pca(index),mse_pca(index)] = computeNMSE(y_test,y_hat);
    [mi_pca(index),bound_pca(index)] = computeMutualInformation(y_test,z_pca_test);
    [mi_x_t_pca(index),~] = computeMutualInformation(z_pca_test,x_test);

    bound_pca(index) = 10*log10(bound_pca(index));
    %GIB test
    A_bn = gib([x_train y_train],nx,ny,nz);
    A_bn = A_bn(1:nz,:);
    z_bn = (x_train*A_bn')+randn(N_train,nz);
    z_bn_test = x_test*A_bn'+randn(N_test,nz);
    
    theta_ls = lmmse("fit","x",z_bn,"y",y_train);
    y_hat = lmmse("predict","x",z_bn_test,"theta",theta_ls);
    
    [nmse_bn(index),mse_bn(index)] = computeNMSE(y_test,y_hat);
    [mi_bn(index),bound_bn(index)] = computeMutualInformation(y_test,z_bn_test);
    [mi_x_t_bn(index),~] = computeMutualInformation(z_bn_test,x_test);
    index = index+1;
end


figure;
plot(n_components_array,10*log10(nmse_bn),'Marker','v','DisplayName','$GIB$','LineWidth',2);
hold on;
plot(n_components_array,10*log10(nmse_pca),'Marker','o','DisplayName','$PCA$','LineWidth',2);
hold on;
l = legend(Location='best');
set(l,'Interpreter','latex',"FontSize",12);
xlim([n_components_array(1),n_components_array(end)]);
set(gca,'XScale','log');
xticks(n_components_array);
grid on;
xlabel("Number of components",'FontSize',14,'Interpreter','latex');
ylabel("NMSE [dB]","FontSize",14,'Interpreter','latex');
title("Comparison of Reconstruction MSE",'Interpreter','latex','FontSize',14);


figure
plot(n_components_array,mi_bn,'Marker','v','DisplayName','GIB','LineWidth',2);
hold on;
plot(n_components_array,mi_pca,'Marker','o','DisplayName','PCA','LineWidth',2);
grid on;
set(gca,'XScale','log');
l = legend(Location='best');
set(l,'Interpreter','latex',"FontSize",12);
xticks(n_components_array);
xlim([n_components_array(1),n_components_array(end)]);
xlabel("Number of components",'FontSize',14,'Interpreter','latex');
ylabel("I(t;y)","FontSize",14,'Interpreter','latex');
title("I(t;y) comparison",'Interpreter','latex','FontSize',14);

figure
plot(n_components_array(1:11),mi_x_t_bn(1:11),'Marker','v','DisplayName','GIB','LineWidth',2);
hold on
plot(n_components_array(1:11),mi_x_t_pca(1:11),'Marker','v','DisplayName','PCA','LineWidth',2)
hold on;
grid on;
title("I(t;x) vs Number of components","FontSize",14,'Interpreter','latex');
xlabel("N. of Components",'FontSize',14,'Interpreter','latex');
ylabel("I(t;x)","FontSize",14,'Interpreter','latex');

figure
plot(n_components_array,10*log10(mse_bn),'Marker','v','DisplayName','M-MMSE','LineWidth',2);
hold on;
plot(n_components_array,10*log10(bound_bn),'Marker','o','DisplayName','$\frac{N_{y}}{(2\pi e)}e^{\frac{2H(y|t)}{N_{y}}}$','LineWidth',2);
grid on;
set(gca,'XScale','log');
xticks(n_components_array);
xlim([n_components_array(1),n_components_array(end)]);
l = legend(Location='best');
set(l,'Interpreter','latex',"FontSize",14);
xlabel("Number of components",'FontSize',14,'Interpreter','latex');
ylabel("MSE","FontSize",14,'Interpreter','latex');



