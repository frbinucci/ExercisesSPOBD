clear all;
close all;

%Setting up the image size
img_width = 32;
img_height = 32;
N = img_width*img_height;

%Image loading
img = imread("peppers.png");
img = rgb2gray(img);
img = imresize(img,[img_width,img_height]);
img = im2double(img);

%Observation Model generation and vectorization
dct_matrix = pinv(dctmtx(img_width));
x = img(:);

psi = kron(dct_matrix,dct_matrix.');

%Plotting the bound function to select the best number of samples
sparsity_axis = 1:N;
bound = sparsity_axis.*log(N./sparsity_axis);

figure;
plot(sparsity_axis,bound,'LineWidth',2);
xlabel("$k$",'FontSize',14,'interpreter','latex');
ylabel("$k\log(\frac{k}{N})$",'FontSize',14,'interpreter','latex');
title("Bound Function, image size = $32\times32$ px",'interpreter','latex','FontSize',14);
grid on;


M = max(bound);
%Let's try the reconstruction with the 70% of samples
M = 0.90*N;
phi = (1/sqrt(M))*randn(int32(M),N);

theta = phi*psi;

y=phi*x;
iterations = 100;
rho = 1;
s_opt = basisPursuitADMM(y,theta,iterations,rho);
rec_img = psi*s_opt;
rec_img = reshape(rec_img,[img_width,img_height]);

psnr_db = computeNormalizedPSNR(img,rec_img);

fprintf("PSNR rec = %3.2f dB\n",psnr_db);

%Showing the reconstructed image

figure
subplot(2,1,1);
imshow(rec_img);
str = strcat("Reconstructed image, PSNR = ",num2str(psnr_db)," dB");
title(str,'FontSize',12,'interpreter','latex');
subplot(2,1,2);
imshow(img);
title("Original Image",'FontSize',12,'interpreter','latex');




%% Part 2: handling bigger images

%Image loading
img_width = 350;
img_height = 350;
rgbImage = imread('peppers.png');
rgbImage = imresize(rgbImage,[img_width img_height]);
bigger_img = im2double(rgb2gray(rgbImage));

%We can consider sub patches with size 10x10px
patch_width = 10;
patch_height = patch_width;
col_patches = int32(img_width/patch_width);
row_patches = int32(img_height/patch_height);
N = patch_width*patch_height;

%Plotting the bound function to select the best number of samples
sparsity_axis = 1:N;
bound = sparsity_axis.*log(N./sparsity_axis);


[value] = max(bound);
%The reconstruction would obey the RIP property for M=37 samples. We can
%thus, we can consider a slightly higher value (e.g., the 55% of the original size) for a good
%reconstruction.
M = 0.55*N;

figure;
plot(sparsity_axis,bound,'LineWidth',2);
xlabel("$k$",'FontSize',14,'interpreter','latex');
ylabel("$k\log(\frac{k}{N})$",'FontSize',14,'interpreter','latex');
title("Bound Function, patch size = $10\times10$ px",'interpreter','latex','FontSize',14);
grid on;

%To handle bigger images we can split the image ind sub-patches.
a = 0;
reverseStr = '';
rec_image = zeros(img_width,img_height);
iterations = 1000;
rho = 1;

%Linear observation model
dct_matrix = pinv(dctmtx(patch_width));
psi = kron(dct_matrix,dct_matrix.');
vectorized_img = bigger_img(:);

for i=1:row_patches
    for j=1:col_patches
        percentDone = 100 * a / (row_patches*col_patches);
        msg = sprintf('Percentage of progress: %3.2f', percentDone); 
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        sub_image = bigger_img((patch_width)*(i-1)+1:(patch_width)*(i-1)+patch_width,(patch_height)*(j-1)+1:(patch_height)*(j-1)+patch_height);
        sub_image = sub_image(:);
        phi = 1/sqrt(M)*randn(int32(M),N);
        theta = phi*psi;
        y = phi*sub_image;
        s_opt = basisPursuitADMM(y,theta,iterations,rho);
        rec_patch = psi*s_opt;
        rec_patch = reshape(rec_patch,[patch_width,patch_width]);
        rec_image((patch_width)*(i-1)+1:(patch_width)*(i-1)+patch_width,(patch_height)*(j-1)+1:(patch_height)*(j-1)+patch_height) = rec_patch;
        a = a+1;
    end
end

% Evaluating the reconstruction PSNR
psnr_db = computeNormalizedPSNR(bigger_img,rec_image);
fprintf("\nPSNR rec = %3.2f dB\n",psnr_db);

% Showing the reconstruted image
figure
subplot(2,1,1);
imshow(rec_image);
str = strcat("Reconstructed image, PSNR = ",num2str(psnr_db)," dB");
title(str,'FontSize',12,'interpreter','latex');
subplot(2,1,2);
imshow(bigger_img);
title("Original Image, size $350 \times 350$ px",'FontSize',12,'interpreter','latex');





















