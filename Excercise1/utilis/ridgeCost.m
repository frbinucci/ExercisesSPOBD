function [cost,fitting_cost,regularization_cost] = ridgeCost(A,x,y,alpha)
%RIDGE_REGRESSION evaluates the cost function of the L2 regularized LS problem
%   Input parameters:
%   A => The design matrix
%   x => The estimated vector
%   y => The observations vector
%   alpha => L1 penalty
%   Outputs:
%   cost => Overall cost function
%   fitting_cost => Fitting cost
%   regularization_cost => Regularization cost
    fitting_cost = (1/2)*sum((A*x-y).^2);
    regularization_cost = alpha*sum(x.*x);
    cost = fitting_cost+regularization_cost;
end

