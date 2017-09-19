% CS 598 PS - ML in Signal Processing
% Problem Set 1 - Problem 2.2.a
% Author: Christian Howard

% load sample image and convert to vectorized grayscale
Ic      = imread('sample_images/test_avg_color.jpg');
Ig      = im2double(rgb2gray(Ic));
[M,N]   = size(Ig);
K       = 1;
vecIc   = im2double(Ic(:));

% construct matrix operator
A = kron(ones(1,K)./K,kron(ones(1,3)./3,kron(eye(N),eye(M))));

% compute average grayscale image across all images and color channels
avg_I = reshape(A*vecIc,M,N);

% show resulting average image
% note this image should be similar (though not necessarily identical) to
% the Ig image above
imshow(avg_I) 
