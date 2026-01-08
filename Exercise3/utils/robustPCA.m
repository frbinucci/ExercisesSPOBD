function [X_opt,A_opt] = robustPCA(Y,lambda,n1,n2)
    cvx_begin quiet
    %cvx_solver mosek
        variables X(n1,n2) A(n1,n2)
        minimize(norm_nuc(X)+lambda*norm(vec(A),1))
        subject to
            Y== X+A;
    cvx_end
    X_opt = X;
    A_opt = A;
end