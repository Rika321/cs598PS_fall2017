import numpy                as np
import matplotlib.pyplot    as plot
import scipy.misc           as misc

import myml.discriminant    as mydsc
import myml.clustering      as mycluster
import hw3.p4_io            as p4io


def attempt3():
    # Author: C. Howard
    # This attempt uses multiple classifiers to distinguish
    # pools from buildings and the 'ground' which includes trees, roads, etc.
    # This means there are going to be three classifiers based on Gaussian
    # models of the dataset.

    # load the training data
    (Xtr,Ltr) = p4io.loadTrainingData2()
    (d,nd) = Xtr.shape

    # define number of classes
    num_classes = 3

    # define the index map
    imap = {'pool':0, 'ground':1, 'buildings':2}

    # build list of discriminant models
    dlist = dict()
    pidx = np.where(Ltr==0)[1]
    Xp = Xtr[:,pidx]
    for k in range(0,num_classes):
        dlist[k] = mydsc.Discriminant(lbl_dataset=Xtr[:,np.where(Ltr==k)[1]],
                                      num_total_data=nd,
                                      Fpinv=np.eye(d,d))

    dlist[imap['pool']].cost_diff       = 2e0
    dlist[imap['ground']].cost_diff     = 1e0
    dlist[imap['buildings']].cost_diff  = 1e0

    # define the testing data
    im_test = misc.imread('ekalismall2.png')
    (nr, nc, d) = im_test.shape
    N = nr * nc
    r = im_test[:, :, 0].reshape(1, N)
    g = im_test[:, :, 1].reshape(1, N)
    b = im_test[:, :, 2].reshape(1, N)

    X = np.zeros((d - 1, N))
    X[0, :] = r
    X[1, :] = g
    X[2, :] = b

    # evaluate testing data based on discriminant models
    test_lbls = mydsc.evalDiscriminantSet(X,discriminant_list=dlist)

    # define colors to visualize the result
    colors=np.array([[255, 125],[0, 125],[0, 125]])

    # obtain new colors based on labels
    #Xn = colors[:,test_lbls]
    num0 = (test_lbls==0).sum()
    T = np.where(test_lbls == 0)
    idx0 = T[0].reshape(1,num0)
    idx1 = np.zeros((1, num0), dtype=int)
    Xn = np.copy(X)
    Xn[:,idx0] = colors[:,idx1]

    # create new image for testing image to show classification result
    im_out = np.copy(im_test)
    for c in range(0,d-1):
        im_out[:,:,c] = Xn[c,:].reshape(nr,nc)

    # generate plot with image
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_out)

    # save the image
    misc.imsave('p4/classified_test_img3.png',im_out)

    # make sure the plot is seen
    plot.show()


def attempt2():
    # Author: C. Howard
    # This attempt uses two classifiers to distinguish
    # pools from everything else (buildings and the 'ground' which includes trees, roads, etc.).
    # This means there are going to be two classifiers based on Gaussian
    # models of the dataset.

    # load the training data
    (Xtr,Ltr) = p4io.loadTrainingData()
    (d,nd) = Xtr.shape

    # define number of classes
    num_classes = 2

    # define the index map
    imap = {'pool':0, 'ground':1}

    # build list of discriminant models
    dlist = dict()
    pidx = np.where(Ltr==0)[1]
    Xp = Xtr[:,pidx]
    for k in range(0,num_classes):
        dlist[k] = mydsc.Discriminant(lbl_dataset=Xtr[:,np.where(Ltr==k)[1]],
                                      num_total_data=nd,
                                      Fpinv=np.eye(d,d))

    # set tuning parameters to emphasize one distribution over another
    dlist[imap['pool']].cost_diff   = 1e-4
    dlist[imap['ground']].cost_diff = 1e0

    # define the testing data
    im_test = misc.imread('ekalismall2.png')
    (nr, nc, d) = im_test.shape
    N = nr * nc
    r = im_test[:, :, 0].reshape(1, N)
    g = im_test[:, :, 1].reshape(1, N)
    b = im_test[:, :, 2].reshape(1, N)

    X = np.zeros((d - 1, N))
    X[0, :] = r
    X[1, :] = g
    X[2, :] = b

    # evaluate testing data based on discriminant models
    test_lbls = mydsc.evalDiscriminantSet(X,discriminant_list=dlist)

    # define colors to visualize the result
    colors=np.array([[255, 125],[0, 125],[0, 125]])

    # obtain new colors based on labels
    #Xn = colors[:,test_lbls]
    num0 = (test_lbls==0).sum()
    T = np.where(test_lbls == 0)
    idx0 = T[0].reshape(1,num0)
    idx1 = np.zeros((1, num0), dtype=int)
    Xn = np.copy(X)
    Xn[:,idx0] = colors[:,idx1]

    # create new image for testing image to show classification result
    im_out = np.copy(im_test)
    for c in range(0,d-1):
        im_out[:,:,c] = Xn[c,:].reshape(nr,nc)

    # generate plot with image
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_out)

    # save the image
    misc.imsave('p4/classified_test_img2.png',im_out)

    # make sure the plot is seen
    plot.show()

def attempt1():
    # Author: C. Howard
    # This attempt uses k-means clustering to identify clusters
    # of similar data in the point cloud. Unfortunately, this did not work well
    # because the pools were such a small fraction of the pixels, I believe k-means
    # treating them more as outliers and smoothing them out with the averaging mechanism.
    #
    # This attempt also tried to use spectral clustering but found the data
    # was too large to perform this using the affinity matrix (not being treated as sparse).
    # This might be achievable if things were treated as sparse or somehow the dataset
    # was shrunk in the number of data points using some clever strategy, but even then it was
    # not clear how this could be used to classify anything. Thus, this first attempt based
    # on clustering was considered inadequate.

    im_train = misc.imread('ekalismall.png')
    im_test = misc.imread('ekalismall2.png')
    (nr, nc, d) = im_train.shape
    N = nr * nc
    r = im_train[:, :, 0].reshape(1, N)
    g = im_train[:, :, 1].reshape(1, N)
    b = im_train[:, :, 2].reshape(1, N)

    X = np.zeros((d - 1, N))
    X[0, :] = r
    X[1, :] = g
    X[2, :] = b

    # (Q,B) = myfac.projrep(X,k_or_tol=2,num_power_method=10)
    # means   = mycluster.kmeans(B,num_means=2,print_msg=True,tol=1e-1)
    # idx     = mycluster.evalKMeans(Q.T@X,means)
    # display the means
    # print(means)
    #
    # # create new image
    # F = Q @ means
    # D = F[:, idx]
    # im_clustered = np.copy(im_train)
    # for c in range(0, 3):
    #     im_clustered[:, :, c] = D[c, :].reshape(nr, nc)

    # perform spectral clustering
    def dist(x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2)

    idx = mycluster.spectral(X, dist, num_means=3, print_msg=True)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

    im_clustered = np.copy(im_train)
    Xn = colors[:, idx]
    for c in range(0, 3):
        im_clustered[:, :, c] = Xn[c, :].reshape(nr, nc)

    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_clustered)
    fig.savefig('p4/clustered_spectral_img.png')

    # fig = plot.figure()
    # ax = mp3d.Axes3D(fig)
    # ax.scatter(r,g,b,cmap=plot.cm.Blues)

    plot.show()