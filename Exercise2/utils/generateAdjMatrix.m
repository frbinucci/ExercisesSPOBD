function [A] = generateAdjMatrix(n)

A = round(rand(n));
A = triu(A) + triu(A,1)';
A = A - diag(diag(A));

end

