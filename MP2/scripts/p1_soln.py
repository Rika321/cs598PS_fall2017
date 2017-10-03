# import useful libraries/modules
import myml.pca     as mypca
import mysp.sound   as mysnd
import myml.images  as myimg
import numpy        as np
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
    Z1 = np.matmul(W1,Sn_d)

    # Perform ICA on lower dimensional measurements
    ica_obj = skd.FastICA(random_state=17)
    ica_obj.fit(Z1.T)
    W2      = np.matmul(ica_obj.components_,W1)
    Z2      = np.matmul(W2,Sn_d)
    W2pi    = np.linalg.pinv(W2)

    # Perform NMF from zero-mean form of data
    nmf_obj = skd.NMF(n_components=num_features, random_state=17)
    H       = nmf_obj.fit_transform(np.abs(Sn_d).T).T
    W3      = nmf_obj.components_.T
    Z3      = H

    # Do the plotting
    ind = 1 + np.arange(0, window_size)

    # plot PCA features
    fg1, ax1 = plot.subplots(nrows = 1, ncols = 3)
    fg1.suptitle('PCA Features')

    for k in range(0,3):
        ax1[k].plot(W1pi[:, k],ind)
        ax1[k].set_title('Feature '+str(k+1))

    fg1.savefig('p1/pca_features_sound.png')

    # plot ICA features
    fg2, ax2 = plot.subplots(nrows=1, ncols=3)
    fg2.suptitle('ICA Features')

    for k in range(0, 3):
        ax2[k].plot(W2pi[:, k], ind)
        ax2[k].set_title('Feature ' + str(k + 1))

    fg2.savefig('p1/ica_features_sound.png')


    # plot NMF features
    fg3, ax3 = plot.subplots(nrows=1, ncols=3)
    fg3.suptitle('NMF Features')

    for k in range(0, 3):
        ax3[k].plot(W3[:, k], ind)
        ax3[k].set_title('Feature ' + str(k + 1))

    fg3.savefig('p1/nmf_features_sound.png')

    # plot PCA weights
    fg4 = plot.figure()
    fg4.suptitle('PCA Weights')
    (fg4,ax4) = myimg.plotReductionWeights(fg4,Z1)
    fg4.savefig('p1/pca_weights_sound.png')

    # plot ICA weights
    fg5 = plot.figure()
    fg5.suptitle('ICA Weights')
    (fg5, ax5) = myimg.plotReductionWeights(fg5, Z2)
    fg5.savefig('p1/ica_weights_sound.png')

    # plot NMF weights
    fg6 = plot.figure()
    fg6.suptitle('PCA Weights')
    (fg6, ax6) = myimg.plotReductionWeights(fg6, Z3)
    fg6.savefig('p1/nmf_weights_sound.png')


    # make sure to show the results
    #plot.show()



