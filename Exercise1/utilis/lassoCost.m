function [cost,fitting_cost,regularization_cost] = lassoCost(A,x,y,alpha)
%lassoCost This function evaluates the lasso cost function
%   Input parameters:
%   A => The design matrix
%   x => The estimated vector
%   y => The observations vector
%   alpha => L1 penalty
%   Outputs:
%   cost => Overall cost function
%   fitting_cost => Fitting cost
%   regularization_cost => Regularization cost
fitting_cost = 1/2*sum(abs(A*x(:,1)-y).^2);
regularization_cost = alpha*sum(abs(x(:,1)));
cost = fitting_cost+regularization_cost;
end

