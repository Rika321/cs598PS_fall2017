function [ A, num_windows ] = genSpectrogramMat( dft_len, hop_size, num_samples )
%genSpectrogramMat - Method to compute matrix that converts signal into
%spectrogram
%   Author  : Christian Howard
%   dft_len : Number of points in Discrete Fourier Transform being used
%   hop_size: Amount of samples window shifts after a window is processed
%   num_samples: Number of samples in complete time series data

% compute number of windows that will be processed
num_windows = fix( (num_samples - hop_size) / (dft_len - hop_size) );

% compute Hann window vector and diagonalize it into a matrix H
w = 0.5.*(1 - cos(2*pi.*((1:dft_len)-1)./(dft_len-1)));
H = diag(w);

% define relative indices
ind = 0:(dft_len-1);

% compute coefficient used in Fourier basis
c = 2*pi/dft_len;

% compute Fourier matrix
D = exp(i.*c.*(ind'*ind))./sqrt(dft_len);

% compute overall processing matrix that processes a local window of data
B = D*H;

% construct the overall transformation matrix that converts a signal into a
% vectorized spectrogram 

% since matrix can be large, use a sparse matrix (this can be slow to form)
% allocate variables that will be used to form sparse matrix in most
% efficient manner
[C,R]   = meshgrid(ind,ind);
rows    = zeros(num_windows,1);
cols    = zeros(num_windows,1);
one_1   = ones(dft_len^2,1);
one_2   = ones(num_windows,1);

% compute top left indices for starting locations of 
% each B submatrix 
for k = 1:num_windows
    rows(k) = 1 + (k-1)*dft_len;
    cols(k) = 1 + (k-1)*hop_size;
end

% use kronecker product to create appropriate vectorized data for
% sparse matrix construction (this is time consuming for lots of data)
row_idx = kron(rows,one_1) + kron(one_2,R(:));
col_idx = kron(cols,one_1) + kron(one_2,C(:));
values  = kron(one_2,B(:));

% construct sparse matrix (time consuming for large amounts of data)
A = sparse(row_idx,col_idx,values);

end

