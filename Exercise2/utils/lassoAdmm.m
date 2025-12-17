function [opt,x,J,fitting_term,regularization_term,C] = lassoAdmm(A,y,alpha,rho,iterations,ref)

    N = size(A,1);
    M = size(A,2);
    x = zeros(M,iterations);
    z=zeros(M,iterations);
    lambda=zeros(M,iterations);
    C = zeros(1,iterations-1);
    J = zeros(1,iterations-1);
    fitting_term = zeros(1,iterations-1);
    regularization_term = zeros(1,iterations-1);
    
    fitting_term(1,:)= 1/2*sum(abs(A*x(:,1)-y).^2);
    regularization_term(1,:) = alpha*sum(abs(x(:,1)));
    J(1,:) = fitting_term+regularization_term;

    for t=1:iterations-1
       if exist('ref','var')
           C(t)=norm(x(:,t)-ref)^2;
       end 
       x(:,t+1)=pinv(A'*A+rho*eye(M))*(A'*y+rho*(z(:,t)-lambda(:,t)));
       z(:,t+1)=wthresh(x(:,t+1)+lambda(:,t),'s',alpha/rho);
       lambda(:,t+1)=lambda(:,t)+x(:,t+1)-z(:,t+1);

       fitting_cost = 1/2*sum(abs(A*x(:,t+1)-y).^2);
       regularization_cost = alpha*sum(abs(x(:,t+1)));
       fitting_term(:,t+1) = fitting_cost;
       regularization_term(:,t+1) = regularization_cost/alpha;
    end

    opt = x(:,iterations);
end