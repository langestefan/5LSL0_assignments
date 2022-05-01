import numpy as np
import matplotlib   as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift


# compute SGD algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*(r_yx - R_x*w[k])
def steepest_gradient_descent(R_x, r_yx, alpha, max_iter, w_init):
    """ 
    :param np.array(3, 3) R_x: autocorrelation matrix
    :param np.array r_yx(3, 1): cross-correlation vector
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w_init(3, 1): initial weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T
    w = w_init

    # compute sgd
    for i in range(max_iter - 1):
        # update weights
        w = w + 2*alpha*(r_yx - np.dot(R_x, w))

        # store weights
        w_history[[i+1], :] = w.T

    return w_history

# compute Newton's method for N iterations
# w[k+1] = w[k] + 2*alpha*(R_x^-1)*(r_yx - R_x*w[k])
def newtons_method(R_x, r_yx, alpha, max_iter, w_init):
    """ 
    :param np.array(3, 3) R_x: autocorrelation matrix
    :param np.array r_yx(3, 1): cross-correlation vector
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w_init(3, 1): initial weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T
    w = w_init

    # compute R_x^-1
    R_x_inv = np.linalg.inv(R_x)

    # compute newtons method
    for i in range(max_iter - 1):
        # update weights
        w = w + 2*alpha*np.dot(R_x_inv, (r_yx - np.dot(R_x, w)))

        # store weights
        w_history[[i+1], :] = w.T

    return w_history

# contour plot function
def contour_plot(w0, w1, w_train, J_vals, title, filename, show=False):

    # plot the contour plot
    fig = plt.figure()
    cp = plt.contour(w0, w1, J_vals)
    plt.clabel(cp, inline=1, fontsize=10)
    plt.plot(w_train[:,0], w_train[:,1])
    plt.title(title)
    plt.xlabel('w0')
    plt.ylabel('w1')

    # plot optimal weights as point in contourplot
    plt.plot(0.2, 1, 'x', color='red', markersize=10)

    if show:
        plt.show()

    # save figure as png
    figure_name = f"linear_models/figures/{filename}.png"
    fig.savefig(figure_name, dpi=300)

if __name__ == "__main__":
    print(" ---- start ---- ")

    # Load the data
    data = pd.read_csv("linear_models/assignment1_data.csv")

    # split data into x and y
    data_x = np.array(data.iloc[:, 0])
    data_y = np.array(data.iloc[:, 1])
    x_stack = np.vstack((data_x, shift(data_x, 1), shift(data_x, 2))).T

    # initialize the filter
    w_3_const = -0.5 # for countourplots
    w_init = np.zeros((3, 1))
    N = data.shape[0]
    alpha = 0.12

    # autocorrelation and cross-correlation matrices
    r_yx = np.array([[1, 5.3, -3.9]]).T 
    R_x = np.array([[5, -1, -2],
                    [-1, 5, -1],
                    [-2, -1, 5]])   

    # apply steepest gradient descent
    w_sgd = steepest_gradient_descent(R_x, r_yx, alpha, N, w_init)
    print("w_sgd optimal: ", w_sgd[N-1])

    # Plot the trajectory of the filter coefficients as they evolve, together with a contour
    # plot of the objective function J. 
    w0 = np.linspace(-0.5, 0.5, 100)
    w1 = np.linspace(-0.1, 1.5, 100)
    W0, W1 = np.meshgrid(w0, w1)
    J_vals = np.zeros((len(W0), len(W1)))

    # compute the objective function J for each point in the grid
    for i in range(len(W0)):
        for j in range(len(W1)):
            w_temp = np.array([W0[0, i], W1[j, 0], w_3_const]).T
            J_vals[i, j] =  mean_squared_error(data_y, np.dot(x_stack, w_temp))

    # plot the contour plot for SGD
    contour_plot(W0, W1, w_sgd, J_vals, 
        title=f"SGD Contour Plot for alpha = {alpha}", 
        filename=f"sgd_contour_plot_alpha_{alpha}")

    # apply newtons method
    w_newton = newtons_method(R_x, r_yx, alpha, N, w_init)

    # plot the contour plot for Newton's method
    contour_plot(W0, W1, w_newton, J_vals,
        title=f"Newton's Method Contour Plot for alpha = {alpha}",
        filename=f"newtons_method_contour_plot_alpha_{alpha}")
