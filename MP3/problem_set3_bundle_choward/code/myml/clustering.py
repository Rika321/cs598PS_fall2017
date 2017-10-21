# import important libs
import numpy as np
import scipy as sp
import scipy.sparse
import myml.factorizations as myfac

def evalKMeans(X, means):
    # Author: Christian Howard
    # Code to compute what cluster some input data is associated with based on distance from
    # input means found via k-means

    (d, nd) = X.shape
    idx = np.zeros((1,nd),dtype=int)

    # assign mean to each coordinate
    for k in range(0, nd):
        delta = np.linalg.norm(means - X[:, k].reshape(d,1), axis=0)
        idx[0,k] = np.argmin(delta)

    # return the output index list
    return idx



def spectral(X, distfunc, num_means, max_iter = 1e3, tol = 1e-3, print_msg = False):
    # Author: Christian Howard
    # Method to perform spectral clustering via an Affinity Matrix formulation,
    # given some distance function and the number of clusters you're looking for

    # get dimensions of data
    (d,nd) = X.shape

    # construct affinity matrix
    if print_msg:
        print('Starting construction of affinity matrix')
    A = np.zeros((nd,nd))
    Dv = np.zeros((nd,1))
    for i in range(0,nd):
        for j in range(0,nd):
            if i != j:
                A[i,j] = distfunc(X[:,i],X[:,j])
        Dv[i] = 1.0/np.sqrt(np.sum(A[i,:]))
        if print_msg:
            print('Finished row {0} out of {1} in construction'.format(i+1,nd))

    if print_msg:
        print('Finished Creation of nominal Affinity Matrix')

    # construct normalized affinity matrix
    Dm = sp.sparse.spdiags(Dv,0,nd,nd)
    An = Dm@A@Dm

    if print_msg:
        print('Finished Creation of Normalized Affinity Matrix')

    # do PCA on the affinity matrix
    (Q,Z) = myfac.projrep(An,k_or_tol=d,num_power_method=10)
    W = Q.T

    if print_msg:
        print('Finished dimensionality reduction')

    # do kmeans on the resulting data Z
    means = kmeans(Z,num_means=num_means,max_iter=max_iter,tol=tol,print_msg=print_msg)

    if print_msg:
        print('Finished kmeans')

    # do cluster classification
    return evalKMeans(Z, means)



def kmeans(X, num_means, max_iter = 1e3, tol = 1e-3, print_msg = False):
    # Author: Christian Howard
    # Function to perform k means on some input dataset X

    # initialize kmeans variables
    (d,nd)  = X.shape
    idx0    = np.random.choice(nd,size=num_means,replace=False)
    means   = X[:,idx0]
    means0  = np.copy(means)
    err     = 1e3
    iter    = 0

    # do k means EM algorithm
    indices = np.zeros((1,nd),dtype=int)
    while iter < max_iter and err > tol:

        # assign cluster
        for k in range(0,nd):
            delta       = np.linalg.norm(means - X[:,k].reshape(d,1),axis=0)
            indices[0,k]= np.argmin(delta)

        # update means
        for k in range(0,num_means):
            means[:,k] = np.mean(X[:,np.where(indices==k)[1]],axis=1)

        # compute change in means
        err = np.linalg.norm(means - means0)
        means0[:,:] = means[:,:]

        # print message
        if print_msg:
            print('After {0} iterations, change in means is {1}'.format(iter,err))

        # update iteration count
        iter += 1

    # return the means
    return means