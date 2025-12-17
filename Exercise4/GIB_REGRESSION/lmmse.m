function [returned_value] = lmmse(mode,varargin)

x_exists = find(cellfun(@(x) strcmpi(x, 'x') , varargin));
y_exists = find(cellfun(@(x) strcmpi(x, 'y') , varargin));
theta_ls_exists = find(cellfun(@(x) strcmpi(x, 'theta') , varargin));
sigma_exists = find(cellfun(@(x) strcmpi(x, 'sigma') , varargin));
nx_exists = find(cellfun(@(x) strcmpi(x, 'nx') , varargin));
ny_exists = find(cellfun(@(x) strcmpi(x, 'ny') , varargin));

if x_exists
    x = varargin{x_exists+1};
end
if y_exists 
    y = varargin{y_exists+1};
end
if theta_ls_exists
    theta_ls = varargin{theta_ls_exists+1};
end
if sigma_exists 
    sigma = varargin{sigma_exists+1};
end
if nx_exists
    nx =varargin{nx_exists+1};
end
if ny_exists
    ny =varargin{ny_exists+1};
end


if strcmp(mode,"fit")
    if  sigma_exists
        sigma_xx = sigma(1:nx,1:nx);
        sigma_xy = sigma(1:nx,nx+1:nx+ny);
        sigma_yx = sigma_xy';
    else
        nx = size(x,2);
        ny = size(y,2);
        sigma = cov([x,y]);
        sigma_xx = sigma(1:nx,1:nx);
        sigma_xy = sigma(1:nx,nx+1:nx+ny);
        sigma_yx = sigma_xy';
    end
    theta_ls = sigma_yx*pinv(sigma_xx);
    returned_value = theta_ls;
%     x = [ones(size(x,1),1) x];
%     theta_ls = x\y;
%     returned_value = theta_ls;
else
    if strcmp(mode,"predict")
        %x = [ones(size(x,1),1) x];
        y_hat = x*theta_ls';
        returned_value = y_hat;
    end
end

end

