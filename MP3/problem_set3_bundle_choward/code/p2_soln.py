#
# Author : Christian Howard
# Purpose: Script to tackle Homework 3 Problem 2
#           which is about digit classification


# import useful libraries
import numpy                as np
import matplotlib.pyplot    as plot
import scipy.io             as sio

# import personal libraries
import myml.factorizations  as myfac
import myml.data_separation as mydata
import myml.images          as myimg
import myopt.root_finding   as myrf
import myml.discriminant    as mydsc

if __name__ == "__main__":

    # get the dataset and break it up into the labels and images
    dataset = sio.loadmat('digits-labels.mat')
    labels  = dataset['l']
    images  = dataset['d']

    # separate data into training and testing data sets
    num_train = 100
    (Dtr,Dtt) = mydata.separateClassData(images,labels,
                                         numdata_or_percent_for_training=num_train)

    # define function to perform dimensionality reduction and produce the
    # set of discriminant functions used to classify the digits
    def discriminant_classification(num_lower_dims, getResultingClassifier = False):

        # get the total number of labels
        nlbl  = Dtr['ulbls'].shape[0]

        # compute mean of data and mean-separated data
        Data  = Dtr['net']
        mean  = myfac.getMeanData(Dtr['net'])
        Datan = Data - mean

        # use randomized projection to do PCA
        #(W,Q,k) = myfac.pca(Datan,k_or_tol=num_lower_dims,getSingularVals=False)
        (Q,B) = myfac.projrep(Datan, k_or_tol=num_lower_dims,num_power_method=10)

        # construct list of discriminants for each label
        (d,nd) = Data.shape
        dlist = dict()
        for i in range(0,nlbl):
            D = Dtr[i] - mean
            dlist[i] = mydsc.Discriminant(lbl_dataset=D,num_total_data=nd,Fpinv=Q.T)

        # test the discriminants for each label dataset in the training set
        total_wrong = 0
        for i in range(0,nlbl):

            # get the ith label's training dataset
            D = Dtt[i] - mean
            (d,nd) = D.shape

            # allocate memory for the discriminant function output values
            results = np.zeros((nlbl,nd))

            # generate the discriminant function values for each
            # discriminant function for each training data point
            for j in range(0,nlbl):
                results[j,:] = dlist[j].eval(D)

            # for each data point, find the index of the discriminant
            # that says the data is best represented by its distribution
            max_idx = np.argmax(results, axis=0)

            # loop through the index array and tally up all indices
            # that do not match the value for i since this should
            # be the correct solution
            for j in range(0,nd):
                if max_idx[j] != i:
                    total_wrong += 1

        # get the total number of training data points
        total_train     = Dtt['net'].shape[1]

        # get the percent of classifications that were right
        percent_right   = 100.0 * (1.0 - total_wrong / total_train)

        # return information based on what this function is being used for
        if getResultingClassifier:
            return (dlist, percent_right, total_wrong, total_train, Q)
        else:
            return percent_right

    # define function to be used within the bisection method to find
    # the optimal number of features to hit 90% correct classification
    def bisfunc(k):
        return discriminant_classification(k) - 90.0

    # use the bisection method to find the optimal dimensionality
    (kopt,iter) = myrf.ibisection(func=bisfunc,bounds=(25,50))
    print('The optimal rank is {0}'.format(kopt))

    # generate the classifier using the result from the bisection algorithm
    (dlist, percent_right, total_wrong, total_train, Q) = discriminant_classification(kopt-1,getResultingClassifier=True)

    # plot the features
    im = myimg.plotImageFeatures(Q,img_dims=(28,28),out_grid_layout=(6,6))
    fig = plot.figure()
    ax  = fig.add_subplot(111)
    ax.imshow(im)
    fig.savefig('p2/resulting_features.png')

    # return various information related to the classification
    print('The total wrong is {0} out of {1} training data points'.format(total_wrong,total_train))
    print('The percent right are {0}%%'.format(percent_right))

