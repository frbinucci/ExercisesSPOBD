function [opt,x,J,fitting_term,regularization_term,C,z] = lassoAdmm(A,y,alpha,rho,iterations,ref)
%lassoAdmm This function allows to solve the L1 regularized fitting problem
%through the ADMM algorithm algorithm
%Input Parameters
%   A => The design matrix
%   y => The observation vector
%   alpha => Penalty on the regularization parameter
%   rho => Augmented lagrangian penalty (cf. ADMM formulation)
%   iterations => Number of iterations
%   ref => True vector (optional)
%Outputs:
%   opt => The estimated vector
%   x => The estimated vectors during the different iterations
%   J => Temporal behaviour of the cost function
%   fitting_term => Temporal behaviour of the fitting term
%   regularization_term => Temporal behaviour of the regularization term
    N = size(A,1);
    M = size(A,2);
    x = randn(M,iterations);
    z = zeros(M,iterations);
    lambda = zeros(M,iterations);
    C = zeros(1,iterations);
    J = zeros(1,iterations);
    fitting_term = zeros(1,iterations);
    regularization_term = zeros(1,iterations);
    
    fitting_term(1,:)        = 1/2*sum(abs(A*x(:,1)-y).^2);
    regularization_term(1,:) = alpha*sum(abs(x(:,1)));
    J(1,:)                   = fitting_term+regularization_term;

    for t=1:iterations-1
       if exist('ref','var')
           C(t)      = norm(x(:,t)-ref)^2;
       end 

       x(:,t+1)      = inv(A'*A+rho*eye(M))*(A'*y+rho*(z(:,t)-lambda(:,t)));
       z(:,t+1)      = SoftThresholdingOperator(x(:,t+1)+lambda(:,t),alpha/rho);
       lambda(:,t+1) = lambda(:,t)+x(:,t+1)-z(:,t+1);

       fitting_cost               = 1/2*sum(abs(A*x(:,t+1)-y).^2);
       regularization_cost        = alpha*sum(abs(x(:,t+1)));
       fitting_term(:,t+1)        = fitting_cost;
       regularization_term(:,t+1) = regularization_cost/alpha;
    end

    opt = x(:,iterations);
end