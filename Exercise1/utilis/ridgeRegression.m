function [opt,x,J,fitting_term,regularization_term,C,diff_centralized] = ridgeRegression(A,y,alpha,gamma,iterations,ref)
%RIDGE_REGRESSION allows to solve a regularized fitting problem through L2
%norm employing a gradient descent algorithm

%   Input parameters:
%   A => design matrix;
%   y => observations vector;
%   alpha => regularization parameter;
%   gamma => step size (learning rate);
%   iterations => number of iterations of GD;

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
fitting_term = zeros(iterations,1);
regularization_term = zeros(iterations,1);

[cost,fitting_cost,regularization_cost] = ridgeCost(A,x(:,1),y,alpha);


fitting_term(1) = fitting_cost;
regularization_term(1) = regularization_cost/alpha;
J(1) = cost;

centralized_solution = pinv(A'*A+alpha*eye(M))*A'*y;

if exist('ref','var')
        C(1)=norm(x(:,1)-ref)^2;
end

diff_centralized = zeros(iterations,1);

for t=1:iterations-1
    %Gradient Step
    x(:,t+1) = x(:,t)-gamma*((A'*A+alpha*eye(M))*x(:,t)-A'*y);% (1-gamma*alpha)*x(:,t) + gamma*(A'*(y-A*x(:,t)));% x(:,t) + gamma*(A'*(y-A*x(:,t))-alpha*x(:,t));
    [cost,fitting_cost,regularization_cost] = ridgeCost(A,x(:,t+1),y,alpha);
    J(t+1) = cost;
    regularization_cost = regularization_cost/alpha;
    fitting_term(t) = fitting_cost;
    regularization_term(t) = regularization_cost;
    if exist('ref','var')
        C(t+1)=mean((x(:,t)-ref).^2);
    end

    diff_centralized(t+1) = norm(x(:,t+1)-centralized_solution)^2;
end

opt = x(:,end);

end

