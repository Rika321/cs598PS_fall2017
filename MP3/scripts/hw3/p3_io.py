import os
import numpy                as np
import scipy.io.wavfile     as wav
import mysp.sound           as mys


def loadRecordedData():
    rpath = 'my_sounds'
    mpath = 'my_sounds/epicmusic.wav'
    spath = 'my_sounds/heyjupiter.wav'

    # specify window and shift
    ws      = 1024
    shift   = 512

    # get data for music
    (freq, signal) = wav.read(mpath)
    N = signal.shape[0]
    sm = signal[:N,0].reshape(N, 1)

    # compute the spectrogram for the music
    Sm = np.log(np.abs(mys.spectrogram(sm,ws=ws,hs=shift)))
    (d,n) = Sm.shape

    # get Sm with the right shape
    d2 = d*(int((freq-ws)/shift)+1)
    n2 = int(n*d/d2)
    N2 = n2*d2
    Sm = Sm.reshape(1,d*n)[0,:N2].reshape(d2,n2)


    # get data for speech
    (freq, signal) = wav.read(spath)
    N = signal.shape[0]
    ss = signal[:N,0].reshape(N, 1)

    # compute the spectrogram for the speech recording
    Ss = np.log(np.abs(mys.spectrogram(ss)))
    (d, n) = Ss.shape

    # get Ss with the right shape
    d2 = d * (int((freq - ws) / shift) + 1)
    n2 = int(n * d / d2)
    N2 = n2 * d2
    Ss = Ss.reshape(1, d * n)[0, :N2].reshape(d2, n2)

    # return the resulting spectrograms
    return (Sm,Ss,freq)

def loadProvidedData():

    # Define path to SpeechMusic
    rpath = 'SpeechMusic'

    # specify window and shift
    ws = 1024
    shift = 512

    if os.path.isfile('{0}/S.npy'.format(rpath)):
        Sn = np.load('{0}/S.npy'.format(rpath))
        Ln = np.load('{0}/L.npy'.format(rpath))
        return (Sn,Ln)
    else:
        # music and speech paths
        mpath = 'SpeechMusic/music'
        spath = 'SpeechMusic/speech'

        # load all the sound and shove into a giant spectrogram dataset as well as a label vector
        # get music data
        iter = 0
        num_m = 0
        Sm = np.array([])
        for file in os.listdir(mpath):

            # get the data and reshape it as necessary
            (freq, signal) = wav.read('{0}/{1}'.format(mpath, file))
            N = signal.shape[0]
            signal = signal.reshape(N, 1)

            # compute the spectrogram
            S = np.log(np.abs(mys.spectrogram(signal,ws=ws,hs=shift)))
            (d, n) = S.shape

            # get S with the right shape
            d2 = d * (int((freq - ws) / shift) + 1)
            n2 = int(n * d / d2)
            N2 = n2 * d2
            S  = S.reshape(1, d * n)[0, :N2].reshape(d2, n2)

            # concatenate data to overall spectrogram
            if iter == 0:
                Sm = S
            else:
                Sm = np.concatenate((Sm, S), axis=1)

            # update iteration
            iter += 1

            # update value for number of music freq vectors
            num_m += n2

            # write output message for progress
            print('Collected data for music sound #{0}'.format(iter))

        # get speech data
        iter = 0
        num_s = 0
        Ss = np.array([])
        for file in os.listdir(spath):

            # get the data and reshape it as necessary
            (freq, signal) = wav.read('{0}/{1}'.format(spath, file))
            N = signal.shape[0]
            signal = signal.reshape(N, 1)

            # compute the spectrogram
            S = np.log(np.abs(mys.spectrogram(signal,ws=ws,hs=shift)))
            (d, n) = S.shape

            # get S with the right shape
            d2 = d * (int((freq - ws) / shift) + 1)
            n2 = int(n * d / d2)
            N2 = n2 * d2
            S = S.reshape(1, d * n)[0, :N2].reshape(d2, n2)

            # concatenate data to overall spectrogram
            if iter == 0:
                Ss = S
            else:
                Ss = np.concatenate((Ss, S), axis=1)

            # update iteration
            iter += 1

            # update value for number of music freq vectors
            num_s += n2

            # write output message for progress
            print('Collected data for speech sound #{0}'.format(iter))

        # Create the labels vector
        Sn = np.concatenate((Sm, Ss), axis=1)
        Ln = np.zeros((1, num_m + num_s))
        Ln[0, num_m:] = 1

        # save the chunks of data
        np.save('{0}/S.npy'.format(rpath),Sn)
        np.save('{0}/L.npy'.format(rpath),Ln)

        # return the chunks of data
        return (Sn,Ln)