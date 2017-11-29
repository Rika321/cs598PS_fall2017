# Author : Christian Howard
# Purpose: To tackle Problem 2 in Homework 2

# import useful libraries/modules
import myml.factorizations      as myfac
import myml.images              as myimg
import numpy                    as np
import scipy.io                 as sio
import matplotlib.pyplot        as plot
import sklearn.decomposition    as skd

import numpy as np
def randproj(A, k, random_seed = 17, num_power_method = 5):
    # Author: Christian Howard
    # This function is designed to take some input matrix A
    # and approximate it by the low-rank form A = Q*(Q^T*A) = Q*B.
    # This form is achieved using randomized algorithms.
    #
    # Inputs:
    # A: Matrix to be approximated by low-rank form
    # k: The target rank the algorithm will strive for.
    # random_seed: The random seed used in code so things are repeatable.
    # num_power_method: Number of power method iterations
    # set the random seed
    np.random.seed(seed=random_seed)
    # get dimensions of A
    (r, c) = A.shape
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
    # return the two matrices
    return Q


if __name__ == '__main__':

    # load image dataset
    digits_mat = sio.loadmat('digits.mat')

    # get the image dataset
    D = digits_mat['d']
    (d, num_images) = D.shape
    print(D.shape)

    # define the number of features for this problem
    num_features = 36

    # Random digit subset
    idx = np.random.choice(num_images,size=49,replace=False)
    Dt = D[:,idx[:]]

    Q = randproj(D,num_features)
    

    # plot the random digits
    fg1 = plot.figure()
    ax1 = fg1.add_subplot(111)
    img_set1 = myimg.plotImageFeatures(Dt, (28, 28), (7, 7))
    im  = ax1.imshow(img_set1.T)
    fg1.colorbar(im)
    fg1.savefig('p2/random_digits.png')