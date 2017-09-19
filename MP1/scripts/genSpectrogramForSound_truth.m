
% define variables used in spectrogram
dft_len  = 1024;
hop_size = 512;

% load the sound
[y,Fs]          = audioread('hello_clip2.m4a');
num_raw_data    = length(y(:,1));
time_elapsed    = num_raw_data/Fs;
sound_data      = y(1:(num_raw_data - mod(num_raw_data,dft_len)),1);
num_data        = length(sound_data);

% generate appropriate spectrogram matrix
S = spectrogram(sound_data,dft_len,hop_size,dft_len,Fs);


%% compute spectrogram
Z = 20.*log10(abs(S));
[r,c] = size(Z);

% setup variables for plotting
x       = linspace(0,1,c).*time_elapsed;
y       = (Fs/dft_len).*(0:(r-1));
[X,Y]   = meshgrid(x,y);

figure
surf(X,Y,Z,'EdgeColor','none','LineStyle','none','FaceLighting','phong'); 
colorbar;
view([0 90])
axis tight
xlabel('Time')
ylabel('Frequency')