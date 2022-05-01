import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift


# compute LMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*x[k]e[k]
def Least_Mean_Square(x_stack, y, alpha, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """

    # compute lms
    for i in range(1, max_iter):
        weight = w[i-1].T
        x_k = x_stack[i].T
        y_hat = np.dot(weight.T,x_k)
        error = data_y[i]-y_hat
        #update weights
        weight_new = weight + 2*alpha*np.dot(x_k,error)
        #stroe weight
        w[i] = weight_new   

    return w

# =============================================================================
# # compute NLMS algorithm for N iterations  
# # w[k+1] = w[k] + 2*alpha/ (x_var^2) *x[k]e[k]
# def Normlized_LMS(R_x, r_yx, alpha, max_iter, w_init):
#     """ 
#     :param np.array(3, 3) R_x: autocorrelation matrix
#     :param np.array r_yx(3, 1): cross-correlation vector
#     :param float alpha: learning rate
#     :param int max_iter: number of iterations
#     :param np.array w_init(3, 1): initial weight vector
#     """
#     # initialize weights
#     w = w_init
# 
#     # initialize weights history
#     w_history = np.zeros((max_iter, 3))
# 
#     # compute R_x^-1
#     R_x_inv = np.linalg.inv(R_x)
# 
#     # compute newtons method
#     for i in range(max_iter):
#         # update weights
#         w = w + 2*alpha*np.dot(R_x_inv, (r_yx - np.dot(R_x, w)))
# 
#         # store weights
#         w_history[[i], :] = w.T
# 
#     return w_history
# =============================================================================

# contour plot function
def contour_plot(w0, w1, w_train, J_vals, title, filename):

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
    plt.show()


    # save figure as png
    figure_name = f"figures/{filename}.png"
    fig.savefig(figure_name, dpi=300)



if __name__ == "__main__":
    print(" ---- start ---- ")

    # Load the data
    data = pd.read_csv("assignment1_data.csv", header=None)

    # split data into x and y
    data_x = np.array(data.iloc[:, 0])
    data_y = np.array(data.iloc[:, 1])
    x_stack = np.vstack((data_x, shift(data_x, 1), shift(data_x, 2))).T

    # initialize the filter
    w_3_const = -0.5 # for countourplots    
    N = data.shape[0]
    w = np.zeros((N, 3))
    alpha = 0.0001
    
        
    # apply mean square error
    w_lms = Least_Mean_Square(x_stack, data_y, alpha, N, w)

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

    # plot the contour plot for LMS
    contour_plot(W0, W1, w_lms, J_vals, 
        title=f"LMS Contour Plot for alpha = {alpha}", 
        filename=f"LMS_contour_plot_alpha_{alpha}")

