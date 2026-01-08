function [X_opt,A_opt] = lowRankPlusSparsityMinimizer(Y,Zpi,R,Pi_m,lambda,n1,n2)
    F = n1;
    T = n2;
    cvx_begin
    %cvx_solver mosek
        variables X(F,T) A(F,T)
        minimize(norm_nuc(X)+lambda*norm(vec(A),1))
        subject to
            R*(X+A)==Y;
            Pi_m.*(X+A)==Zpi;
    cvx_end
    X_opt = X;
    A_opt = A;
end

