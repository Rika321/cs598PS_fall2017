% CS 598 PS - ML in Signal Processing
% Problem Set 1 - Problem 2.2.b
% Author: Christian Howard

% load sample image and convert to vectorized grayscale
Ic      = imread('sample_images/test_avg_color.jpg');
Ig      = im2double(rgb2gray(Ic));
[M,N]   = size(Ig);
K       = 1;
vecIc   = im2double(Ic(:));

% construct matrix operator
A = kron(ones(1,K)./K,kron([1, 0, 0],kron(eye(N),eye(M))));

% compute average red channel image
avg_r = reshape(A*vecIc,M,N);

% show average image (should be all white for the current test image)
imshow(avg_r)