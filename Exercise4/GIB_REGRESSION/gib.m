function [A] = gib(data,nx,ny,n_comp)
    
    sigma = cov(data);
    sigma_xx = sigma(1:nx,1:nx);
    sigma_yy = sigma(nx+1:nx+ny,nx+1:nx+ny);
    sigma_xy = sigma(1:nx,nx+1:nx+ny);
    sigma_yx = sigma_xy';
    
    gib_matrix = eye(nx)-sigma_xy*pinv(sigma_yy)*sigma_yx*pinv(sigma_xx);

    A = zeros(nx,nx);

    [rigt_eigv,lambda,left_eigv] = eig(gib_matrix);
    left_eigv = real(left_eigv);
    lambda(lambda<0) = 0;
    lambda(lambda>1) = 1;
    lambda = diag(real(abs(lambda)));
    [lambda, ind] = sort(lambda);
    left_eigv = left_eigv(:,ind);

    
    beta = 1./(1-lambda);
    if n_comp==nx
        bc = beta(n_comp);
    else
        bc = beta(n_comp+1);
    end
    n_comp = 0;
    
    for i=1:numel(beta)
        if beta(i)<=bc
            ri = left_eigv(:,i)'*sigma_xx*left_eigv(:,i);
            arg_sq = (bc*(1-lambda(i))-1)/(lambda(i)*ri);
            A(i,:) =sqrt(arg_sq)*left_eigv(:,i);
            n_comp = n_comp + 1;
        end
    end
end

