#import useful libraries
import numpy as np
import math

def spectrogram(signal, ws = 1024, hs = 512):
    # Author: C. Howard
    # Function to compute the baseline complex valued spectrogram for some input sound signal.
    # signal: a sound represented as a column vector
    # ws    : the window size
    # hs    : the hope size, aka the amount we shift from sample to sample

    # compute Hamming weights
    alpha   = 0.54
    beta    = 1 - alpha
    pi2     = 2.0*math.pi
    c       = pi2/(ws-1)
    w       = alpha - beta * np.cos(c*np.arange(0,ws,1))

    # compute DFT matrix
    p = np.arange(0,ws,1).reshape(ws,1)
    F = np.exp( -(pi2/ws)*np.matmul(p,p.T)*1j )  / np.sqrt(ws)

    # compute resulting local matrix
    D = np.multiply(F,w)

    # Compute number of samples in spectrogram
    (len,c)     = signal.shape
    num_samples = math.floor((len - hs)/(ws - hs))

    # initialize output S
    S = np.zeros((ws,num_samples),dtype=type(F[0,0]))

    # loop through and construct S
    for i in range(0,num_samples):
        S[:ws, i] = np.matmul(D,signal[i*hs:(i*hs+ws),0])

    # return the output spectrogram
    return S

