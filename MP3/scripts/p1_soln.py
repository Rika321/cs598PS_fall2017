#
# Author : Christian Howard
# Purpose: Script to tackle Homework 3 Problem 1
#           which is about Discriminants


# import useful libraries
import numpy                as np
import matplotlib.pyplot    as plot
import myml.discriminant    as dscmt



if __name__ == "__main__":

    # define the means/covariances and stuff them into a list
    # set 1
    m11 = np.zeros((2,1))
    C11 = np.eye(2,2)
    m21 = np.array([4,0]).reshape(2,1)
    C21 = np.eye(2,2)

    #set 2
    m12 = np.zeros((2,1))
    C12 = np.diag([1,2])
    m22 = np.array([4,3]).reshape(2,1)
    C22 = np.eye(2,2)

    #set 3
    m13 = np.zeros((2,1))
    C13 = np.diag([1,2])
    m23 = np.array([0.5,0]).reshape(2,1)
    C23 = np.eye(2,2)

    #set 4
    m14 = np.zeros((2,1))
    C14 = np.diag([1,2])
    m24 = np.array([4,0]).reshape(2,1)
    C24 = np.diag([2,1])

    # clump all the items into a set of tuples
    m1 = [m11, m12, m13, m14]
    C1 = [C11, C12, C13, C14]
    m2 = [m21, m22, m23, m24]
    C2 = [C21, C22, C23, C24]

    # setup plotting related variables
    N   = 500
    x1  = np.linspace(-3,7,N)
    x2  = np.linspace(-4,6,N)
    (X1,X2) = np.meshgrid(x1, x2)
    U = np.concatenate((X1.reshape(1,N*N),X2.reshape(1,N*N)),0)
    figures = [plot.figure(),plot.figure(),plot.figure(),plot.figure()]
    plot.rc('text', usetex=True)

    # loop through each set of means and covariances and do the necessary work
    for (part, mean1, cov1, mean2, cov2, fig) in zip(['a','b','c','d'],m1,C1,m2,C2, figures):
        gaussDiscr1 = dscmt.getGaussianDiscriminantParams(mean1, cov1, 0.5)
        gaussDiscr2 = dscmt.getGaussianDiscriminantParams(mean2, cov2, 0.5)
        Z = (dscmt.evalGaussianDiscriminant(U,gaussDiscr1)
                    - dscmt.evalGaussianDiscriminant(U,gaussDiscr2)).reshape(N,N)

        # plot boundary
        ax = fig.add_subplot(111)
        h0 = ax.contour(X1,X2,Z,cmap=plot.get_cmap('viridis'),levels=[0])
        plot.clabel(h0,inline=1, fontsize=10)

        # plot the gaussian distributions
        G1 = dscmt.evalGuassianPDF(U, mean1, cov1, gaussDiscr1[3]).reshape(N, N)
        G2 = dscmt.evalGuassianPDF(U, mean2, cov2, gaussDiscr2[3]).reshape(N, N)
        h1 = ax.contour(X1,X2,G1,cmap=plot.get_cmap('plasma'))
        plot.clabel(h1, inline=1, fontsize=10)
        h2 = ax.contour(X1,X2,G2,cmap=plot.get_cmap('plasma'))
        plot.clabel(h2, inline=1, fontsize=10)
        ax.set_xlabel('$$x_1$$')
        ax.set_ylabel('$$x_2$$')
        #ax.set_title('Part {0} Discriminant'.format(part))
        ax.axis('equal')

        # save the figure
        fig.savefig('p1/discriminant_{0}.png'.format(part),dpi=300)




