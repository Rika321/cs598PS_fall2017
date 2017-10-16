#
# Author : Christian Howard
# Purpose: Script to tackle Homework 3 Problem 4
#           which is about classification of pools


# import useful libraries
import os
import numpy                as np
import matplotlib.pyplot    as plot
import mpl_toolkits.mplot3d as mp3d
import scipy.io             as sio
import scipy.io.wavfile     as wav
import sklearn.decomposition    as skd
import scipy.misc           as misc

# import personal libraries
import mysp.sound           as mys
import myml.factorizations  as myfac
import myml.data_separation as mydata
import myml.images          as myimg
import myopt.root_finding   as myrf
import myml.discriminant    as mydsc
import hw3.p3_io            as p3io
import myml.clustering      as mycluster
import hw3.p4soln           as p4soln

if __name__ == "__main__":

    # run attempt at solving problem 4 in hw 3
    p4soln.attempt2()
