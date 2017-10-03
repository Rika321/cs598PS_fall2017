# import important libs
import numpy as np
import numpy.linalg as la


# method to perform PCA on some input data given the dimensionality
# we want our resulting features to have or given a tolerance on the
# ratio between singular values and the max singular value
def pca( D, k_or_tol ):

    # perform economy SVD on input data
    [U,S,Vt] = la.svd(D, full_matrices = False )

    # make copy of singular values
    iv = np.copy(S)

    # define current number of weight terms as follow
    k = k_or_tol

    # if we are picking k using a relative difference
    # between singular value magnitudes
    if k_or_tol < 1:
        b = iv >= S[0]*k_or_tol  # find indices for all singular value ratios > threshold
        k = np.sum(b)

    # compute weight matrix and pseudo-inverse of weight matrix using truncated SVD terms
    W       = U[:, :k].T
    Winv    = U[:, :k]

    # return weights and original singular values
    return (W,Winv,S)



