function [y] = SoftThresholdingOperator(x,beta)
    y = sign(x).*( max( 0, abs(x)-beta) );
end

