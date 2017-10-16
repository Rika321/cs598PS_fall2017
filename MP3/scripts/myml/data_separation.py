import numpy as np

def separateClassData( X, Y, numdata_or_percent_for_training ):
    # Method to separate data into training and testing sets
    # This method does this by getting either the number of data
    # for training or the percent of data you want for training

    # get parameter for number/percent of training data
    (d,Nd) = X.shape
    Ntr = numdata_or_percent_for_training
    if Ntr <= 1:
        Ntr = int(Ntr*Nd)
    else:
        Ntr = int(Ntr)

    # get the unique labels
    list_y = Y.tolist()[0]
    ulbl = np.array(sorted(list(set(list_y))))
    (nlbl,) = ulbl.shape

    # get subsets of data for training and testing
    Train   = dict()
    Test    = dict()
    Train['ulbls']  = ulbl
    Test['ulbls']   = ulbl
    Train['net']    = np.zeros((d, Ntr*nlbl))
    Test['net']     = np.zeros((d, Nd - Ntr*nlbl))
    Train['nlbl']   = np.zeros((1, Ntr * nlbl))
    Test['nlbl']    = np.zeros((1, Nd-Ntr*nlbl))

    sidx = 0
    for idx,label in zip(range(0,nlbl),ulbl):
        (idxv,) = np.where(Y[0,:] == label)
        (nld,)  = idxv.shape
        Ntt     = nld - Ntr
        Train[label]= X[:,idxv[:Ntr]]
        Test[label] = X[:,idxv[Ntr:]]
        Train['net'][:,(idx*Ntr):((idx+1)*Ntr)] = X[:, idxv[:Ntr]]
        Test['net'][:,sidx:(sidx+Ntt)]          = X[:, idxv[Ntr:]]
        Train['nlbl'][0,(idx*Ntr):((idx+1)*Ntr)]= Y[0, idxv[:Ntr]]
        Test['nlbl'][0,sidx:(sidx+Ntt)]         = Y[0, idxv[Ntr:]]
        sidx += Ntt

    # return the resulting training and testing data sets
    return (Train, Test)
