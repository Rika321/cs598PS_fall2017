import myml.pca     as mypca
import mysp.sound   as mysnd
import numpy        as np
import numpy.linalg as la
import scipy.io.wavfile as wav
import matplotlib.pyplot   as plot
import sklearn.decomposition as skd

if __name__ == '__main__':

    # define the number of features you want
    num_features    = 3

    # load the sound file and get into proper shape
    (freq,signal)   = wav.read('vl1.wav')
    N               = signal.shape[0]
    signal          = signal.reshape(N,1)

    # produce the spectrogram of the sound signal
    window_size     = 1024;
    hop_size        = int(window_size/4)
    S               = mysnd.spectrogram(signal,ws=window_size,hs=hop_size)

    # find the sqrt of the magnitude of the spectrogram matrix
    (r,c)           = S.shape
    Sn              = np.sqrt(np.abs(S))
    #Sn              = np.abs(S)

    # compute mean of spectrogram data
    mu_s            = np.matmul(Sn,np.ones((c,1))/c)

    # subtract mean from data
    Sn_d            = Sn - mu_s

    # Perform PCA from zero-mean form of data
    (W1,W1pi, sv)   = mypca.pca(Sn_d,num_features)

    # compute lower dimensional measurements
    Z = np.matmul(W1,Sn_d)

    # Perform ICA on lower dimensional measurements
    ica_obj = skd.FastICA(random_state=17)
    ica_obj.fit(Z.T)
    W2      = np.matmul(ica_obj.components_,W1)
    W2pi    = np.linalg.pinv(W2)

    # Perform NMF from zero-mean form of data
    nmf_obj = skd.NMF(n_components=num_features, random_state=17)
    H       = nmf_obj.fit_transform(Sn.T).T
    W3      = nmf_obj.components_.T

    # Do the plotting
    ind = 1 + np.arange(0, window_size)

    # plot PCA result
    fg1 = plot.figure()
    ax1 = fg1.add_subplot(111)
    ax1.plot(ind, W1pi[:, 0], ind, W1pi[:, 1], ind, W1pi[:, 2])
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Feature Component Magnitude')

    # plot ICA result
    fg2 = plot.figure()
    ax2 = fg2.add_subplot(111)
    ax2.plot(ind, W2pi[:, 2], ind, W2pi[:, 0], ind, W2pi[:, 1])
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Feature Component Magnitude')

    # plot ICA result
    fg3 = plot.figure()
    ax3 = fg3.add_subplot(111)
    ax3.plot(ind, W3[:, 0], ind, W3[:, 1], ind, W3[:, 2])
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Feature Component Magnitude')

    # make sure to show the results
    plot.show()



