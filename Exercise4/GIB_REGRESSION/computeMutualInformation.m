function [mi_x_y,bound] = computeMutualInformation(X,Y)

nx = size(X,2);
ny = size(Y,2);
sigma = cov([X Y]);

sigma_xx = sigma(1:nx,1:nx);
sigma_yy = sigma(nx+1:nx+ny,nx+1:nx+ny);
sigma_xy = sigma(1:nx,nx+1:nx+ny);
sigma_yx = sigma_xy';

sigma_xx_cond_y = sigma_xx-sigma_xy*pinv(sigma_yy)*sigma_yx;

h_x = 0.5*log(((2*pi*exp(1))^nx)*det(sigma_xx));
h_x_cond_y = real(0.5*log(((2*pi*exp(1))^nx)*det(sigma_xx_cond_y)));

mi_x_y = h_x-h_x_cond_y;
bound = (nx/(2*pi*exp(1)))*exp(2*h_x_cond_y/nx);

end

