# import important libs
import numpy as np
import numpy.linalg as la
import math

def getMeanData( D ):
    (d,nd) = D.shape
    mv = np.ones((nd,1))/nd
    mean = D@mv
    return mean

def pca( D, k_or_tol , getSingularVals = True):
    # Author: C. Howard
    # method to perform PCA on some input data given the dimensionality
    # we want our resulting features to have or given a tolerance on the
    # ratio between singular values and the max singular value

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
    if getSingularVals:
        return (W,Winv,k,S)
    return (W,Winv,k)

def ica( D , eps = 1e-6, random_seed = 17, useStandard=True):
    # Author: C. Howard
    # method to perform ICA using FastICA on some input data to create a set
    # of independent features
    #
    # Inputs:
    #   D           : Dataset where columns represent samples and
    #                 rows are number of dimensions
    #   eps         : the epsilon used to denote when the algorithm has converged
    #   random_seed : seed to use to make algorithm repeatable
    #
    # Outputs:
    #   W   : Mixing Matrix
    #   Winv: Inverse of Mixing Matrix
    #
    # Code based on https://en.wikipedia.org/wiki/FastICA and comments made in
    # Machine Learning a Probabilistic Perspective by Kevin P. Murphy

    # initialize random seed for repeatability
    np.random.seed(seed=random_seed)

    # define the g(u) and g'(u) that will be used
    def g_gderiv1(u):
        c = np.exp(-0.5*u**2)
        return (u*c, (1 - u**2)*c)

    def g_gderiv2(u):
        c = np.tanh(u)
        return (c, 1 - c**2)

    if useStandard:
        g_gderiv = g_gderiv2
    else:
        g_gderiv = g_gderiv1

    # get number of components that will be used
    (N,M) = D.shape
    ncomp = N

    # define output matrix W
    W = np.random.randn(D.shape[0],ncomp)

    # define some helper entities
    One = np.ones((M,1))

    # compute the components
    for p in range(0,ncomp):

        # set difference between w vectors to arbitrary number above threshold
        dw = 10*eps

        while dw > eps:

            # get current value for vector w
            w0 = np.copy(W[:,p])

            # compute source signal
            z = np.matmul(W[:,p].T,D)

            # compute g and g' given z
            (g,gd) = g_gderiv(z)

            # set the new value for the vector w
            W[:,p] = (np.matmul(D,g.T) - np.matmul(gd,One)*W[:,p])/M

            # orthogonalize relative to the past w components
            if p > 0:
                W[:,p] = W[:,p] - np.matmul( W[:,:p], (np.matmul(W[:,p].T,W[:,:p])) )

            # normalize to keep on constraint surface
            W[:,p]  = W[:,p]/np.sqrt(np.matmul(W[:,p].T,W[:,p]))

            # compute change in w vector
            dw = np.linalg.norm(w0-W[:,p])

    # Compute inverse of W
    Winv = W.T

    # return W and inverse of W
    return (W,Winv)


def projrep(A, k_or_tol, use_tol = False, return_k = False, random_seed = 17, num_power_method = 5, num_adapt_kstep = 7):
    # Author: Christian Howard
    # This function is designed to take some input matrix A
    # and approximate it by the low-rank form A = Q*(Q^T*A) = Q*B.
    # This form is achieved using randomized algorithms and
    # allows for adaptive rank reduction
    #
    # Inputs:
    # A: Matrix to be approximated by low-rank form
    # k_or_tol: If value >= 1, it sets that rank as the target.
    #           If 0 < value < 1, tries to adaptively find low rank form

    # set the random seed
    np.random.seed(seed=random_seed)

    # get dimensions of A
    (r, c) = A.shape

    # get the smallest dimension
    sdim = min(r, c)

    if not use_tol and k_or_tol >= 1:
        k = int(k_or_tol)

        # get the random input and measurements from column space
        omega   = np.random.randn(c, k)
        Y       = np.matmul(A, omega)

        # form estimate for Q using power method
        for i in range(1, num_power_method):
            Q1, R1 = np.linalg.qr(Y)
            Q2, R2 = np.linalg.qr(np.matmul(A.T, Q1[:, :k]))
            Y = np.matmul(A, Q2[:, :k])
        Q3, R3 = np.linalg.qr(Y)

        # get final k orthogonal vector estimates from column space
        Q = Q3[:, :k]

        # compute weights associated with the column space to approximate A
        B = np.matmul(Q.T,A)

        # return the two matrices
        if return_k:
            return (Q,B,k)
        else:
            return (Q,B)
    else:

        # init some variables used in algorithm
        eps = k_or_tol
        err = 1e20
        kt  = 0
        k   = num_adapt_kstep
        Qt  = np.zeros((r, c))
        At  = np.copy(A)
        iter= 1
        Q   = np.zeros((r,k))

        # adaptively try to estimate the rank via
        # finding column space estimates
        while err > eps and kt < (sdim-k):

            # get the random input and measurements
            omega = np.random.randn(c, k)
            Y = np.matmul(At, omega)

            # form estimate for Q using power method
            for i in range(1, num_power_method):
                Q1, R1 = np.linalg.qr(Y)
                Q2, R2 = np.linalg.qr(np.matmul(At.T, Q1[:, :k]))
                Y = np.matmul(At, Q2[:, :k])
            Q3, R3 = np.linalg.qr(Y)
            Q = Q3[:, :k]  # get final k orthogonal vector estimates from column space

            # compute normalized eigenvector estimate for error matrix E
            z = np.matmul(At, omega[:,0])
            y = z - np.matmul(Q, np.matmul(Q.T, z))
            y = y / np.linalg.norm(y)

            # use multiple iterations of power method y = (E*E')^{n}*y
            # to get most dominant singular vector
            for i in range(0, num_power_method):
                z = y - np.matmul(Q, np.matmul(Q.T, y))
                y = np.matmul(At.T,z)
                y = y / np.linalg.norm(y)
                z = np.matmul(At, y)
                y = z - np.matmul(Q, np.matmul(Q.T, z))
                y = y / np.linalg.norm(y)

            # compute largest singular value estimate based on dominant singular vector
            # assume singular value estimate as the error since same as 2-norm of E
            z = y - np.matmul(Q, np.matmul(Q.T, y))
            t = np.matmul(At.T, z)
            err = (t.T@t) / (y.T@y)



            # Stitch new Q vectors to net Q vectors
            Qt[:, kt:(kt + k)] = Q

            # update total k value
            kt      = kt + k
            iter    += 1

            # orthogonalize vectors using gram schmidt
            #if iter != 1:
            #    Qh, Rh = np.linalg.qr(Qt[:, :kt])
            #    Qt[:, :kt] = Qh
            if iter != 1:
                s = kt - k
                for i in range(s,kt):
                    for j in range(0,i-1):
                        Qt[:,i] = Qt[:,i] - (np.dot(Qt[:,i],Qt[:,j])/np.dot(Qt[:,j],Qt[:,j]))*Qt[:,j]
                    Qt[:,i] = Qt[:,i]/np.sqrt(np.dot(Qt[:,i],Qt[:,i]))

            # retreive modified k-recent Q vectors
            Q = Qt[:, (kt - k):kt]

            # update matrix A based on new Q vectors
            # Note that new A matrices are essentially error matrices
            # since we subtract approximate projection representations of the kth A matrix
            At = At - np.matmul(Q, np.matmul(Q.T, At))

        # set the final Q matrix
        Q = Qt[:, :kt]

        # compute weights associated with the column space to approximate A
        B = np.matmul(Q.T, A)

        # return the two matrices
        if return_k:
            return (Q,B,kt)
        else:
            return (Q,B)


