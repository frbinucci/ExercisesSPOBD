function [psnr_db] = computeNormalizedPSNR(y1,y2)
%computeNormalizedPSNR Computation of the normalized PSNR between two
%matrices
%   y1 => Ground truth matrix
%   y2 => Reconstructed/noisy matrix

mI = max(max(y1));
mse = norm(y1-y2,"fro")^2/norm(y1,"fro")^2;

psnr_db = 20*log10(mI/sqrt(mse));

end

