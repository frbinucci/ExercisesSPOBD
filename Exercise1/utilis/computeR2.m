function [r2] = computeR2(y,y_hat)
    %computeR2 This function computes the R2 score as metric to evaluate
    %the quality of a linear regression fitting: https://en.wikipedia.org/wiki/Coefficient_of_determination
    %Input Parameters:
    %   y => The ground truth vector
    %   y_hat => The estimated vector
    %Outputs:
    %   r2 => The computed R2 score
    rss = sum((y-y_hat).^2);
    y_line = mean(y);
    tss = sum((y-y_line).^2);
    r2 = 1 - rss/tss;
end

