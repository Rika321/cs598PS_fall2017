import numpy as np
import math
import myml.factorizations as myfac

def getGaussianDiscriminantParams(mean, covariance,ProbOmega):
    # Author: Christian Howard
    # Function to get Gaussian discriminant function parameters given
    # some mean, covariance, and probability of some class

    invC = np.linalg.inv(covariance)
    Wm = -0.5*invC
    Wv = invC@mean
    ws = -0.5*(mean.T@Wv) - 0.5*np.log(np.linalg.det(covariance)) + np.log(ProbOmega)
    return (Wm, Wv, ws, invC)

def evalGaussianDiscriminant(x, discrParams ):
    # Author: Christian Howard
    # Function to compute a gaussian discriminant given some input X
    # and a tuple with the discriminant parameters

    Wm = discrParams[0]
    Wv = discrParams[1]
    ws = discrParams[2]
    return (x.T.dot(Wm)*x.T).sum(axis=1) + Wv.T.dot(x) + ws

def evalGuassianPDF(x, mean, cov, inv_cov):
    # Author: Christian Howard
    # Function to evaluate a Gaussian PDF in any dimension given the
    # necessary hyperparameters

    delta = x - mean
    return np.sqrt(np.linalg.det((2.0*math.pi)*cov))*np.exp( -0.5*(delta.T.dot(inv_cov)*delta.T).sum(axis=1) )

def evalDiscriminantSet(X, discriminant_list):
    # Author: Christian Howard
    # Function to compute the classification given some input X
    # and a list of discriminant functions that could label the
    # input data

    # get the total number of labels
    nlbl = len(discriminant_list)

    # get dimensions of data
    (d,nd) = X.shape

    # init matrix for evaluating discriminants against data
    results = np.zeros((nlbl, nd))

    # loop discriminant functions and figure out
    # the classification for each data point in X
    for i in range(0, nlbl):
        results[i, :] = discriminant_list[i].eval(X)

    # for each data point, find the index of the discriminant
    # that says the data is best represented by its distribution
    max_idx = np.argmax(results, axis=0)

    # return the max_idx which represents
    # the classification for each data point in X
    return max_idx


class Discriminant:
    # Author: Christian Howard
    # Class representing a Gaussian discriminant function to help
    # simplify creating and evaluating them

    def __init__(self):
        self.Fpinv      = []
        self.Wm         = []
        self.Wv         = []
        self.Ws         = []
        self.cost_diff  = 1.0

    def __init__(self, lbl_dataset, num_total_data, Fpinv, cost_diff=1.0):
        self.Fpinv = Fpinv
        (d,nd)  = lbl_dataset.shape
        ldataset= Fpinv@lbl_dataset
        mean    = myfac.getMeanData(ldataset)
        Wsigma  = ldataset - mean
        cov     = (Wsigma@Wsigma.T)/(nd-1.0)
        covinv  = np.linalg.inv(cov)
        Pomega  = nd/num_total_data
        self.Ws = np.log(Pomega) - 0.5*np.log(np.linalg.det(cov)) - 0.5*mean.T@(covinv@mean)
        self.Wv = (mean.T@covinv).T
        self.Wm = -0.5*covinv
        self.cost_diff = cost_diff

    def eval(self,x):
        w = self.Fpinv@x
        return np.log(self.cost_diff) + evalGaussianDiscriminant(w,(self.Wm, self.Wv, self.Ws))
