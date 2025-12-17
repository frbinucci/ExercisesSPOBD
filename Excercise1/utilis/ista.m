function [opt,x,J,fitting_term,regularization_term,C] = ista(A,y,alpha,gamma,iterations,ref)
%ista This function allows to solve the L1 regularized fitting problem
%through the ista algorithm
%Input Parameters
%   A => The design matrix
%   y => The observation vector
%   alpha => Penalty on the regularization parameter
%   gamma => Learning rate
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
    C = zeros(1,iterations);
    J = zeros(1,iterations);
    fitting_term = zeros(1,iterations);
    regularization_term = zeros(1,iterations);
    
    [cost,fitting_cost,regularization_cost] = lassoCost(A,x(:,1),y,alpha);
    
    regularization_term(1) = regularization_cost/alpha;
    fitting_term(1) = fitting_cost;
    J(1) = cost;
    if exist('ref','var')
            C(1)=norm(x(:,1)-ref)^2;
    end
    
    for t=1:iterations-1
        gradientStep = x(:,t)+gamma*A'*(y-A*x(:,t));
        x(:,t+1) = SoftThresholdingOperator(gradientStep,alpha*gamma);
        fitting_cost = 1/2*sum(abs(A*x(:,t+1)-y).^2);
        regularization_cost = alpha*sum(abs(x(:,t+1)));
        fitting_term(:,t+1) = fitting_cost;
        regularization_term(:,t+1) = regularization_cost/alpha;
        J(:,t+1) =  fitting_cost+ regularization_cost;
        if exist('ref','var')
            C(t+1)=norm(x(:,t)-ref)^2;
        end
    end
    
    opt = x(:,end);
end
