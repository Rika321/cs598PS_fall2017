import numpy as np
import math

def getGaussianDiscriminantParams(mean, covariance,ProbOmega):
    invC = np.linalg.inv(covariance)
    Wm = -0.5*invC
    Wv = invC@mean
    ws = -0.5*(mean.T@Wv) - 0.5*np.log(np.linalg.det(covariance)) + np.log(ProbOmega)
    return (Wm, Wv, ws, invC)

def evalGaussianDiscriminant(x, discrParams ):
    Wm = discrParams[0]
    Wv = discrParams[1]
    ws = discrParams[2]
    return (x.T.dot(Wm)*x.T).sum(axis=1) + Wv.T@x + ws

def evalGuassianPDF(x, mean, cov, inv_cov):
    delta = x - mean
    return np.sqrt(np.linalg.det((2.0*math.pi)*cov))*np.exp( -0.5*(delta.T.dot(inv_cov)*delta.T).sum(axis=1) )