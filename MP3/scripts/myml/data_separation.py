import numpy as np

def separateClassData( X, Y, numdata_or_percent_for_training ):
    # Author: Christian Howard
    # Method to separate data into training and testing sets
    # This method does this by getting either the number of data
    # for training or the percent of data you want for training

    # get parameter for number/percent of training data
    (d,Nd) = X.shape
    Ntr = numdata_or_percent_for_training
    frac= numdata_or_percent_for_training
    if Ntr <= 1:
        Ntr = int(Ntr*Nd)
    else:
        frac = float(Ntr) / float(Nd)
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
    Train['net']    = np.array([])
    Test['net']     = np.array([])
    Train['nlbl']   = np.array([])
    Test['nlbl']    = np.array([])

    sitr = 0
    sitt = 0

    for idx,label in zip(range(0,nlbl),ulbl):
        (idxv,) = np.where(Y[0,:] == label)
        (nld,)  = idxv.shape
        if numdata_or_percent_for_training <= 1:
            nltr    = int(frac*nld)
        else:
            nltr    = numdata_or_percent_for_training

        (idx1, idx2)= np.split(idxv,[nltr])

        Train[label]= X[:,idx1]
        Test[label] = X[:,idx2]

        if idx != 0:
            Train['net']    = np.concatenate( (X[:, idx1],Train['net']), axis=1 )
            Test['net']     = np.concatenate( (X[:, idx2],Test['net']), axis=1 )
            Train['nlbl']   = np.concatenate( (Y[:, idx1],Train['nlbl']), axis=1 )
            Test['nlbl']    = np.concatenate( (Y[:, idx2],Test['nlbl']), axis=1 )
        else:
            Train['net']    = X[:, idx1]
            Test['net']     = X[:, idx2]
            Train['nlbl']   = Y[:, idx1]
            Test['nlbl']    = Y[:, idx2]

    # return the resulting training and testing data sets
    return (Train, Test)
