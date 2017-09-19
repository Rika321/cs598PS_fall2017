% CS 598 PS - ML in Signal Processing
% Problem Set 1 - Problem 2.1.b
% Author: Christian Howard

% load sample image and convert to vectorized grayscale
Ic      = imread('sample_images/test_avg_color.jpg');
Ig      = im2double(rgb2gray(Ic));
[M,N]   = size(Ig);
vecIg   = real(Ig(:));

% construct matrix operator
A = kron(ones(1,N)./N,kron(eye(2), 2/(M).*ones(1,M/2)));

% compute average colors
avg = A*vecIg;

