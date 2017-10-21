#
# Author : Christian Howard
# Purpose: Script to tackle Homework 3 Problem 3
#           which is about classification between
#           voices and music


# import useful libraries
import os
import numpy                as np
import matplotlib.pyplot    as plot

# import personal libraries
import myml.factorizations  as myfac
import myml.data_separation as mydata
import myml.discriminant    as mydsc
import myml.images          as myimg
import hw3.p3_io            as p3io

if __name__ == "__main__":

    # load recorded sound spectrograms
    (Sm, Ss, Sm0, Ss0, freq) = p3io.loadRecordedData()

    # get music/speech dataset
    # frequency is at 22,050 Hz
    (S,L) = p3io.loadProvidedData()
    (d,nd)= S.shape
    smean = myfac.getMeanData(S)

    # create map for the music and speech indices
    imap = {'music':0, 'speech':1}

    # get the lower dimensional features for the dataset
    (Q,Z) = myfac.projrep(S,k_or_tol=4,num_power_method=10)
    W=Q.T

    # break up the dataset into a training and testing dataset a couple times
    # and evaluate the performance of the discriminant classifier developed
    ind = np.linspace(0,nd-1,dtype=int)
    num_experiments = 1000
    num_classes     = 2
    success_rate = np.zeros((num_experiments,1))
    for ne in range(0,num_experiments):

        # shuffle indices
        np.random.shuffle(ind)

        # create new dataset based on shuffling
        Sn = S[:,ind]
        Ln = L[:,ind]

        # Split dataset into training and testing
        (train, test) = mydata.separateClassData( Sn, Ln, numdata_or_percent_for_training=0.90 )

        # create the classifier
        dlist = dict()
        dlist[imap['music']]  = mydsc.Discriminant(lbl_dataset=train[imap['music']],  # music discriminant
                                                   num_total_data=nd,
                                                   Fpinv=W)
        dlist[imap['speech']] = mydsc.Discriminant(lbl_dataset=train[imap['speech']], # speech discriminant
                                                   num_total_data=nd,
                                                   Fpinv=W)

        # evaluate the classifier at each subset of data in testing set
        for i in range(0,num_classes):
            D = test[i]
            classification = mydsc.evalDiscriminantSet(X=D,discriminant_list=dlist)
            for idx in classification:
                if idx == i:
                    success_rate[ne] += 1.0

        # compute resulting success rate for the nth experiment
        success_rate[ne] = success_rate[ne] / test['net'].shape[1]

    # The average success rate
    print('The average testing success rate was {0}%'.format(100.0 * np.mean(success_rate)))

    # train the classifier on all the dataset
    dlist   = dict()
    ind0    = np.where(L == 0)[1]
    ind1    = np.where(L == 1)[1]
    dlist[imap['music']] = mydsc.Discriminant(lbl_dataset=S[:,ind0],  # music discriminant
                                              num_total_data=nd,
                                              Fpinv=W)
    dlist[imap['speech']] = mydsc.Discriminant(lbl_dataset=S[:,ind1],  # speech discriminant
                                               num_total_data=nd,
                                               Fpinv=W)


    # evaluate the music recording, using the first second worth of spectrogram data
    X = Sm[:,0].reshape(d,1)
    classes = mydsc.evalDiscriminantSet(X,dlist)
    print('The average class is {0}'.format(np.mean(classes)))
    expected_class = np.round(np.mean(classes))

    if expected_class == 0:
        print('Recorded music is classified as Music class')
    else:
        print('Recorded music is classified as Speech class')

    # evaluate the music recording, using the first second worth of spectrogram data
    X = Ss[:,0].reshape(d,1)
    classes = mydsc.evalDiscriminantSet(X, dlist)
    print('The average class is {0}'.format(np.mean(classes)))
    expected_class = np.round(np.mean(classes))

    if expected_class == 0:
        print('Recorded speech is classified as Music class')
    else:
        print('Recorded speech is classified as Speech class')

    # generate spectrogram images for input sounds
    f1 = plot.figure()
    (f1, ax1) = myimg.plotDataset(f1, Ss0)
    ax1.set_xlabel('Number of Time Samples')
    ax1.set_ylabel('Number of Frequency Terms')
    f1.savefig('p3/spectrogram_speech.png')

    f2 = plot.figure()
    (f2, ax2) = myimg.plotDataset(f2, Sm0)
    ax2.set_xlabel('Number of Time Samples')
    ax2.set_ylabel('Number of Frequency Terms')
    f2.savefig('p3/spectrogram_music.png')


