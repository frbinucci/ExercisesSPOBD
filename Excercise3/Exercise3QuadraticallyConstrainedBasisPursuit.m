clear all;
close all;

%Image loading and noise corruption
image_width = 128;
image_height = 128;

rgbImage = imread('peppers.png');
rgbImage = imresize(rgbImage,[image_width image_height]);
grayImage= im2double(rgb2gray(rgbImage));

power = sum(sum(grayImage.*grayImage))/(size(grayImage,1)*size(grayImage,2));
snr_db = 10;
snr = 10^(snr_db/10);
pnoise = power/snr;
noisyImage = grayImage + sqrt(pnoise)*randn(size(grayImage));

patch_size = 16;
n_patches = ((image_width*image_height)/(patch_size)^2);
n_rows = image_width/patch_size;
n_cols = image_height/patch_size;

psnr = computeNormalizedPSNR(grayImage,noisyImage);

M = floor(patch_size^2);
phi = randn(M,patch_size^2);

reconstructed_image = zeros(image_width,image_height);
dct_matrix = dctmtx(patch_size);
theta = kron(pinv(dct_matrix),pinv(dct_matrix'));
a = 0;
reverseStr = '';

%MSE constraint
epsilon = 20;

for i=1:n_rows
    for j=1:n_cols
        percentDone = 100 * (a / (n_rows*n_cols));
        msg = sprintf('Percentage of progress: %3.1f', percentDone); 
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        vectorized_image = reshape(noisyImage((i-1)*patch_size+1:i*patch_size,(j-1)*patch_size+1:(j)*patch_size),[],1);
        vectorized_image = phi*vectorized_image;
        cvx_begin quiet
        variable s(patch_size^2)
        minimize norm(s,1)
        subject to
        norm(vectorized_image-phi*theta*s,2)<=epsilon;
        cvx_end
        reconstructed_patch = theta*s;
        reconstructed_patch = reshape(reconstructed_patch,[patch_size,patch_size]);
        reconstructed_image((i-1)*patch_size+1:i*patch_size,(j-1)*patch_size+1:(j)*patch_size)=reconstructed_patch;
        a = a+1;
    end
end

%% Evaluating the reconstruction PSNR.
psnr_rec = computeNormalizedPSNR(grayImage,reconstructed_image);

%Plotting the noisy and the de-noised image
figure
subplot(2,1,1);
imshow(noisyImage);
title(strcat("Noisy Image, PSNR =",num2str(psnr)," dB"),'interpreter','latex','FontSize',14);
subplot(2,1,2);
imshow(reconstructed_image);
title(strcat("De-noised Image, PSNR =",num2str(psnr_rec)," dB"),'interpreter','latex','FontSize',14);

fprintf("\n\n");
fprintf("The PSNR of the original image is %f\n",psnr);
fprintf("The PSNR of the reconstructed image is %f",psnr_rec);