# **Fall 2017** - CS 598 PS: Machine Learning in Signal Processing

## Purpose for Repo
This repository acts a location to place CS 598 PS Homeworks (MPs) and related documents/reports for the Fall 2017 course taught by [Paris Smaragdis](http://paris.smaragd.is/).

## Course Description
The goal of this course was to be a fairly intensive survey course, with a good mix of theory and practical insights, in Machine Learning applied to areas of Signal Processing. This course covered many Machine Learning topics in dimensionality reduction, classification, regression, clustering, latent variable models. We then took these techniques and looked into solving problems involving audio, images, video, other time series. These problems ranges from classification problems to regression problems, estimating missing data, and more.

## Problem Sets
### Problem Set 1
This problem set was really more of a basic mathematical and signal processing primer. It covered some areas of probability theory, tensor math, and some work related to spectrograms. The report can be found at: 

### Problem Set 2
Within this problem set, a big goal was to use unsupervised learning techniques, such as PCA, ICA, and NMF, to investigated dimensionality reductions on some audio and computer vision related problems. For some of the image data, some manifold learning techniques were done to see if there was some underlying, nonlinear structure in some handwritten digit image datasets. 

### Problem Set 3
This problem set was pretty heavily revolved around classification problems. The first problem revolved around some fundamental theory to Gaussian Discriminant Functions. The second problem revolved around using Gaussian Discriminant Functions to classify digits. The third problem was to take a satellite image of some area of homes, some with pools, to come up with a pool detector. The approach was open to individuals to figure out. I experimented with some clustering techniques, like k-means and affinity matrix based clustering, but found these were not too useful since pools were treated as outliers relative to the other pixels in the image. The resulting approach ended up being to mark the image with three classes, one for pools, one for the ground, and one for the roads. Using this labeled data, a Gaussian Discriminant Classifier was constructed to classify parts of the image and was used to produce a fairly robust pool detector.

## Final Project
The final project repository is located at: [https://github.com/choward1491/cs598ps_project](https://github.com/choward1491/cs598ps_project). This project revolved around using dimensionality reduction and classification techniques to process and understand fMRI data labeled according to people with depression and people without depression. Various classification models were applied and all found great classification success, think 95%+, using a reduced dimensionality of about 20.
