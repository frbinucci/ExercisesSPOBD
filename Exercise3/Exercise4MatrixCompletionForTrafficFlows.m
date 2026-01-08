clear all;
close all;
%Setting up a generic seed for reproducibility
rng(1);
%Data loading
load SAND_TM_Estimation_Data.mat;

%Data of ALL the traffic flows
Q1=X'; 
%Let us select a subset of the whole set of temporal indexes
Q=Q1(:,1:50); 
%Routing matrix (PAY attention to the notation!)
R=A; 
clear A;

%Aggregated traffic
Y=R*Q; 

%Sampling probability
perc=0.2; 

%Number of links
L=size(Y,1); 
%Number of flows
F=size(Q,1); 
%Time instants (50)
T=size(Q,2);  

N=F*T;

m=round(perc*N);
Pi_m=zeros(F,T);
samples=randsample(N,m);
Pi_m(samples)=1;
%Flow level measurements (we assume to have access to the 20% of flows)
Zpi=Pi_m.*(Q); 

%Solving the optimization problem...
lambda = 0.2;
[X_opt,A_opt] = lowRankPlusSparsityMinimizer(Y,Zpi,R,Pi_m,lambda,F,T);
%%
%Evalutation metrics
MSE=10*log10((norm(Q-(X_opt+A_opt),"fro")^2)/(norm(Q,"fro")^2));

fprintf("MSE reconstructed traffic = %3.2f dB",MSE);
%%
% Plotting of traffic matrices
figure
subplot(1,4,1);imagesc(Q); title('True Traffic Matrix')
subplot(1,4,2);imagesc(X_opt+A_opt); title('Recovered Traffic Matrix')
subplot(1,4,3);imagesc(X_opt); title('Low-rank Component')
subplot(1,4,4);imagesc(A_opt); title('Sparse Component')
colormap(gray)

% Plotting the temporal behavior of a generic flow (e.g., 70).
ind=70;
figure
subplot(3,1,1);plot(1:T,Q(ind,:),'-.b','Linewidth',2.5);
hold on; plot(1:T,X_opt(ind,:)+A_opt(ind,:),'-.g','Linewidth',2.5); legend('True traffic','Estimated traffic + anomalies')
xlabel('Time index','FontSize',14)
ylabel('Traffic Level','FontSize',14)
ylim([-1e7 4e7])
grid on
subplot(3,1,2);plot(1:T,X_opt(ind,:),'-k','Linewidth',2.5); legend('Estimated clean traffic')
xlabel('Time index')
ylabel('Traffic Level')
grid on
subplot(3,1,3);plot(1:T,A_opt(ind,:),'-r','Linewidth',2.5); legend('Estimated anomalies')
xlabel('Time index','FontSize',14)
ylabel('Anomaly Level','FontSize',14)
grid on
ylim([-1e7 3e7])
