# import useful libraries/modules
import myml.pca                 as mypca
import myml.images              as myimg
import numpy                    as np
import scipy.io                 as sio
import matplotlib.pyplot        as plot
import sklearn.manifold         as skm


if __name__ == '__main__':

    # load image dataset
    digits_mat = sio.loadmat('digits.mat')

    # get the image dataset
    D0      = digits_mat['d']
    labels  = digits_mat['l']
    ind     = np.where(labels == 6)
    D       = D0[:,ind[1]]
    (d, num_images) = D.shape

    # define the number of features for this problem
    num_features = 2

    # compute mean of images
    mu_i = np.matmul(D, np.ones((num_images, 1)) / num_images)

    # subtract mean from data
    Dn = D - mu_i

    # Perform PCA from zero-mean form of data
    (W1, W1pi, sv) = mypca.pca(Dn, num_features)

    # compute lower dimensional measurements
    Z = np.matmul(W1, Dn)

    # plot the feature sets for PCA
    fg1 = plot.figure()
    ax1 = fg1.add_subplot(111)
    img_set1 = myimg.plot_img_features(W1pi, (28, 28), (1, 2))
    ax1.imshow(img_set1)
    ax1.set_title('Two Dominant Features for digit 6')
    fg1.savefig('p3/pca_features_6.png')

    # use latex
    plot.rc('text', usetex=True)

    # plot the weights with images associated with the weight
    fg2 = plot.figure()
    (fg2,ax2) = myimg.plotImagesInWeightSpace(fg2,Z,Dn,(28,28))

    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('PCA - Original Images for digit 6 in Weight Space')
    fg2.savefig('p3/pca_weightspace_images_6.png',dpi=300)


    # Compute the Laplacian Eigenmap
    se = skm.SpectralEmbedding(n_components=2, random_state=17,n_neighbors=10)
    se.fit(Dn.T)
    Ze = se.embedding_.T

    # plot the eigenmap result
    fg3 = plot.figure()
    (fg3, ax3) = myimg.plotImagesInWeightSpace(fg3, Ze, Dn, (28, 28))

    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('Laplacian Map - Original Images for digit 6 in Weight Space')
    fg3.savefig('p3/selm_weightspace_images_6.png', dpi=300)


