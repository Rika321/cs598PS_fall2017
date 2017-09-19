% CS 598 PS - ML in Signal Processing
% Problem Set 1 - Problem 3
% Author: Christian Howard

%% clear the workspace
clear all; close all;

%% define variables used in spectrogram
dft_len  = 1024;
hop_size = 512;
num_data = 4*dft_len;

%% generate appropriate spectrogram matrix
[A, num_windows] = genSpectrogramMat(dft_len, hop_size, num_data);

%% plot spectrogram matrix magnitude
figure
surf(abs(A),'EdgeColor','none','LineStyle','none','FaceLighting','phong')
view([0 90])
axis tight
title('Spectrogram Matrix','FontSize',16)
xlabel('Columns','FontSize',16)
ylabel('Rows','FontSize',16)
set(gca,'Ydir','reverse')

%% save the images
print(gcf,'-dpng','-r300',['png/spectrogram_matrix.png'])
saveas(gcf,['fig/spectrogram_matrix.fig'])