clear all;
close all;

% Image loading
img_width = 256;
img_height = 256;
y = im2double(rgb2gray(imresize(imread('concorde.jpg'),[img_width,img_height])));

% Add salt-and-pepper noise
noise_density = 0.15;  
y_noise = imnoise(y, 'salt & pepper', noise_density);
 
n1 = size(y,1);
n2 = size(y,2);

N=n1*n2;

patch_width = 64;
patch_height = 64;

%Setting up the regolarization parameter (you can also try different values
%and choose the one that leads to the best reconstruction).
lambda = 1 / sqrt(max(patch_width, patch_height));

patch_size = [patch_width,patch_height];

% Define row and column sizes for mat2cell
rows = img_height/patch_height;
cols = img_width/patch_width;

% Image reconstruction
noise_components = zeros(img_width,img_height);
reconstructed = zeros(img_width,img_height);
for i=1:rows
    for j=1:cols
        sub_image = y_noise((patch_width)*(i-1)+1:(patch_width)*(i-1)+patch_width,(patch_height)*(j-1)+1:(patch_height)*(j-1)+patch_height);
        [X,A] = robustPCA(sub_image,lambda,patch_width,patch_height);
        reconstructed((patch_width)*(i-1)+1:(patch_width)*(i-1)+patch_width,(patch_height)*(j-1)+1:(patch_height)*(j-1)+patch_height) = X;
        noise_components((patch_width)*(i-1)+1:(patch_width)*(i-1)+patch_width,(patch_height)*(j-1)+1:(patch_height)*(j-1)+patch_height) = A;
    end
end

%Performance evaluation
noisy_PSNR = computeNormalizedPSNR(y,y_noise);
reconstructed_PSNR = computeNormalizedPSNR(y,reconstructed);

%% Showing the de-noised image.
figure
subplot(2,2,1);
imshow(y_noise);
title(strcat("Noisy, PSNR=",num2str(noisy_PSNR)," dB"),'FontSize',14,'Interpreter','latex');
subplot(2,2,2);
imshow(y);
title("Original Image","FontSize",14,'Interpreter','latex');
subplot(2,2,3);
imshow(noise_components);
title("Noise Component",'FontSize',14,'Interpreter','latex');
subplot(2,2,4);
imshow(reconstructed)
title(strcat("Rec., PSNR=",num2str(reconstructed_PSNR)," dB"),'FontSize',14,'Interpreter','latex');
fprintf("PSNR noisy signal: %f dB\n",noisy_PSNR);
fprintf("PSNR reconstructed signal: %f dB",reconstructed_PSNR);