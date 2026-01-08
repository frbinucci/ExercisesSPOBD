clear all;
close all;

%Setting a Random Seed for reproducibility
rng(46);

%Setting up the image size
img_width = 32;
img_height = 32;
N = img_width*img_height;

%Loading and transforming the image
img = imread("peppers.png");
img = rgb2gray(img);
img = imresize(img,[img_width,img_height]);
img = im2double(img);

%Corrupting the image with Gaussian noise
snr_db = 10;
snr = 10^(snr_db/10);
signal_power = sum_square(img(:))/numel(img);
p_noise = signal_power/snr;
noisy_img = img+sqrt(p_noise)*randn(size(img));

%Computing the PSNR
psnr = computeNormalizedPSNR(img,noisy_img);
fprintf('Normalized PSNR = %3.2f\n',psnr);

% LASSO image de-noising

%Orthonormal Basis Computation
dct_matrix = pinv(dctmtx(img_width));

%Observation model vectorization
y = noisy_img(:);
psi = kron(dct_matrix,dct_matrix.');

%Setting up a set of sparsity constraints
sparsity_constraints = [5,15,25,35,45,55,65,75,85,95];
psnr_rec = zeros(length(sparsity_constraints),1);

%We want to track the best number of components
best_psnr = 0;
best_rec = zeros(size(img));

for i=1:numel(sparsity_constraints)
    fprintf("Sparsity constraint n. %d\n",i)
    
    cvx_begin quiet
       variable s(N)
       minimize sum_square(y-psi*s)
       subject to
       sum(abs(s))<=sparsity_constraints(i);
    cvx_end
    y_rec = psi*s;
    y_rec = reshape(y_rec,[img_width,img_height]);
    psnr_db = computeNormalizedPSNR(img,y_rec);
    if psnr_db>best_psnr
        best_psnr = psnr_db;
        best_rec = y_rec;
    end
    psnr_rec(i) = psnr_db;
end

%Plotting the PSNR for different sparsity constraints
figure;
semilogx(sparsity_constraints,psnr_rec,'LineWidth',2);
yline(psnr,'LineWidth',2,'LineStyle','--','DisplayName','Original MSE');
xlabel("Sparsity",'FontSize',14);
ylabel('Reconstruction MSE','FontSize',14);
grid on;


%Best reconstructed image rescaling (for visualization purposes)
min_rec = min(min(best_rec));
max_rec = max(max(best_rec));
min_img = min(min(img));
max_img = max(max(img));
best_rec_rescaled = best_rec-min_rec;
best_rec_rescaled = best_rec_rescaled*((max_img-min_img)/(max_rec-min_rec));
best_rec_rescaled = best_rec_rescaled+min_img;

%Plotting the best reconstructed image
figure
subplot(3,1,1);
imshow(noisy_img);
str = strcat("Noisy image, PSNR = ",num2str(psnr)," dB");
title(str,'FontSize',12,'interpreter','latex');
subplot(3,1,2);
imshow(best_rec_rescaled);
str = strcat("De-noised image, PSNR = ",num2str(best_psnr)," dB");
title(str,'FontSize',12,'interpreter','latex');
subplot(3,1,3);
imshow(img);
title("Original Image",'FontSize',12,'interpreter','latex');