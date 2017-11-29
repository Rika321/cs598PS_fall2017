import numpy                as np
from matplotlib.image       import BboxImage
from matplotlib.transforms  import Bbox, TransformedBbox
import matplotlib.pyplot    as plot


def plotImageFeatures(F, img_dims, out_grid_layout):
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

def plotDataset(fig,D):
    ax = fig.add_subplot(111)
    im = ax.imshow(D, interpolation='nearest', cmap=plot.cm.inferno,
                   extent=(0.5, np.shape(D)[1] + 0.5,
                           0.5, np.shape(D)[0] + 0.5))
    ax.set_aspect('auto')
    ax.set_xlabel('Number of Data Points')
    ax.set_ylabel('Number of Feature Dimensions')
    fig.colorbar(im)
    return (fig, ax)

def plotReductionWeights(fig,Z):
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, interpolation='nearest', cmap=plot.cm.ocean,
                    extent=(0.5, np.shape(Z)[1] + 0.5,
                            0.5, np.shape(Z)[0] + 0.5))
    ax.set_aspect('auto')
    ax.set_xlabel('Number of Data Points')
    ax.set_ylabel('Number of Feature Dimensions')
    fig.colorbar(im)
    return (fig,ax)


def plotImagesInWeightSpace( fig, wspace, orig_images, img_dims, margin_percent = 0.03, img_ratio = 0.035 ):
    # Author: C. Howard
    # function to robustly plot images in the weight space
    #
    # fig: Figure handle for plotting
    # wspace        : The lower dimensional features such that wspace = W*X where W maps
    #                 the input data X into lower dim data
    # orig_images   : This is basically X where each column is a vectorized image
    # img_dims      : This is the rectangular dimensions of each image in the data set
    # margin_percent: What percent of the data span we want to add margin in the plot
    # img_ratio     : What ratio we use to decide image sizes to be plotted,  relative to the data span


    # get number of original images
    (d,num_images) = orig_images.shape

    # get max dims of data
    x1max = np.max(wspace[0, :])
    x1min = np.min(wspace[0, :])
    x2max = np.max(wspace[1, :])
    x2min = np.min(wspace[1, :])

    # get the center
    x1c = 0.5 * (x1max + x1min)
    x2c = 0.5 * (x2max + x2min)

    # get width and height
    w = x1max - x1min
    h = x2max - x2min

    # compute how much to scale the width/height from the
    # center of the data to set the xlimits/ylimits
    scale = 0.5 + margin_percent

    # plot the images in the weight space
    ax = fig.add_subplot(111)

    for k in range(0, num_images):
        bb          = Bbox.from_bounds(wspace[0, k], wspace[1, k], w * img_ratio, h * img_ratio)
        bb2         = TransformedBbox(bb, ax.transData)
        bbox_img    = BboxImage(bb2, norm=None, origin=None, clip_on=False)
        bbox_img.set_data(orig_images[:, k].reshape(img_dims[0], img_dims[0]))
        ax.add_artist(bbox_img)

    # set the axis limits
    ax.set_xlim(x1c - scale * w, x1c + scale * w)
    ax.set_ylim(x2c - scale * h, x2c + scale * h)

    # return the fig handle and axis handle
    return (fig,ax)
