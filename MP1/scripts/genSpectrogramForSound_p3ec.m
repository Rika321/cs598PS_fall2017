% CS 598 PS - ML in Signal Processing
% Problem Set 1 - Problem 3 Extra credit
% Author: Christian Howard

%% clear the workspace
clear all; close all;

%% define variables used in spectrogram
dft_len  = 1024;
hop_size = 512;

%% load the sound
filename        = 'music_clip1';
[y,Fs]          = audioread(['audio/',filename,'.m4a']);
num_raw_data    = length(y(:,1));
time_elapsed    = num_raw_data/Fs;
sound_data      = y(1:(num_raw_data - mod(num_raw_data,dft_len)),1);
num_data        = length(sound_data);

%% generate appropriate spectrogram matrix
[A, num_windows] = genSpectrogramMat(dft_len, hop_size, num_data);

%% compute spectrogram
S       = A*sound_data;
Smag    = 20*log10(abs(S)); % put magnitude in decibels 
Z       = reshape(Smag,dft_len,num_windows);

%% setup variables for plotting
x       = linspace(0,1,num_windows).*time_elapsed;  % time
y       = (Fs/dft_len).*(0:(dft_len-1));            % frequencies
[X,Y]   = meshgrid(x,y);

%% plot the results
figure
surf(X,Y,Z,'EdgeColor','none','LineStyle','none','FaceLighting','phong'); 

% set the right view and tighten axis
view([0 90])
axis tight

% label appropriate items
h = colorbar;
title('Spectrogram','FontSize',16)
ylabel(h, 'Amplitude (dB)','FontSize',16)
xlabel('Time (s)','FontSize',16)
ylabel('Frequency (Hz)','FontSize',16)

%% save the images
print(gcf,'-dpng','-r300',['png/',filename,'_spectrogram.png'])
saveas(gcf,['fig/',filename,'_spectrogram.fig'])