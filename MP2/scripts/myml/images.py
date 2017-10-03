import numpy        as np


def plot_img_features(F, img_dims, out_grid_layout):
    # Author: C. Howard
    # This is a function that takes a matrix of features, F, where features are
    # defined column-wise, and builds an image with all of these feature images
    # so one can look at all the features at once
    #
    # F                 : Matrix of features where each feature is a column
    # img_dims          : Dimensions of an individual feature
    # out_grid_layout   : Grid layout of the output image

    # initialize output image that will comprise of all the image features
    # all in their own space
    out_img = np.zeros((img_dims[0]*out_grid_layout[0],img_dims[1]*out_grid_layout[1]))

    # get dimensions of feature matrix
    (d, num_features)   = F.shape

    # get dimensions of image grid
    (nrows, ncols)      = out_grid_layout

    # get the image dimensions
    (ir,ic)             = img_dims

    # build the output matrix
    for k in range(0,num_features):
        r = int(k / ncols)
        c = int( k % ncols )
        out_img[r*ir:(r+1)*ir,c*ic:(c++1)*img_dims[1]] = F[:,k].reshape(img_dims[0],img_dims[1])


    # return the output image of features
    return out_img

