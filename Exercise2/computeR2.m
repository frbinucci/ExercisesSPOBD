function [r2] = computeR2(y,y_hat)
    rss = sum((y-y_hat).^2);
    y_line = mean(y);
    tss = sum((y-y_line).^2);
    r2 = 1 - rss/tss;
end

