function [s_opt] = basisPursuitADMM(y,theta,iterations,rho)
    
    N = size(theta,2);
    s = zeros(N,iterations);
    u = zeros(N,iterations);
    z = zeros(N,iterations);
    
    pinv_theta = pinv(theta*theta');
    for k=1:iterations-1
        s(:,k+1)=(eye(N)-theta'*pinv_theta*theta)*(z(:,k)-u(:,k))+theta'*pinv_theta*y;
        z(:,k+1)=wthresh(u(:,k)+s(:,k+1),'s',1/rho);
        u(:,k+1)=u(:,k)+s(:,k+1)-z(:,k+1);
    end

    s_opt = s(:,k+1);

end

