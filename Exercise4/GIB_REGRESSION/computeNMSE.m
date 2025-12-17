function [nmse,mse] = computeNMSE(y,y_hat)

nmse = 0;
mse = 0;
N_samples = size(y,1);
for i=1:size(y,1)
    diff = y(i,:)-y_hat(i,:);
    nmse = nmse + sum(diff.*diff)/sum(y(i,:).*y(i,:));
    mse = mse + sum(diff.*diff);
end
nmse = nmse/N_samples;
mse = mse/N_samples;
end

