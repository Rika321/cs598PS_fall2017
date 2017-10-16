import os
import numpy                as np
import scipy.misc           as misc


def loadTrainingData():
    # Define path to training images
    rpath = 'p4_data'

    if os.path.isfile('{0}/X.npy'.format(rpath)):
        Xn = np.load('{0}/X.npy'.format(rpath))
        Ln = np.load('{0}/L.npy'.format(rpath))
        return (Xn,Ln)
    else:
        # pool and ground training data
        gpath = 'p4_data/ground'
        ppath = 'p4_data/pools'

        # load all image data for pools
        iter    = 0
        num_p   = 0
        Xp      = np.array([])
        for file in os.listdir(ppath):

            im = misc.imread('{0}/{1}'.format(ppath,file))
            (nr, nc, d) = im.shape
            N = nr * nc
            r = im[:, :, 0].reshape(1, N)
            g = im[:, :, 1].reshape(1, N)
            b = im[:, :, 2].reshape(1, N)

            X = np.zeros((d - 1, N))
            X[0, :] = r
            X[1, :] = g
            X[2, :] = b

            # concatenate data to overall spectrogram
            if iter == 0:
                Xp = X
            else:
                Xp = np.concatenate((Xp, X), axis=1)

            # update iteration
            iter += 1

            # update value for number of music freq vectors
            num_p += N

        # get data for ground/non-pool image areas
        iter    = 0
        num_g   = 0
        Xg      = np.array([])
        for file in os.listdir(gpath):

            im = misc.imread('{0}/{1}'.format(gpath,file))
            (nr, nc, d) = im.shape
            N = nr * nc
            r = im[:, :, 0].reshape(1, N)
            g = im[:, :, 1].reshape(1, N)
            b = im[:, :, 2].reshape(1, N)

            X = np.zeros((d - 1, N))
            X[0, :] = r
            X[1, :] = g
            X[2, :] = b

            # concatenate data to overall spectrogram
            if iter == 0:
                Xg = X
            else:
                Xg = np.concatenate((Xg, X), axis=1)

            # update iteration
            iter += 1

            # update value for number of music freq vectors
            num_g += N

        # Create the labels vector
        Xn = np.concatenate((Xp, Xg), axis=1)
        Ln = np.zeros((1, num_p + num_g))
        Ln[0, num_p:] = 1

        # save the chunks of data
        np.save('{0}/X.npy'.format(rpath), Xn)
        np.save('{0}/L.npy'.format(rpath), Ln)

        # return the image data and labels
        return (Xn,Ln)
