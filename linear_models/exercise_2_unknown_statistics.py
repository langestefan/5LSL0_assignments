import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.ndimage import shift


# compute LMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*x[k]e[k]
def Least_Mean_Square(x_stack, y, alpha, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """

    # compute lms
    for i in range(1, max_iter):
        weight = w[i-1].T
        x_k = x_stack[i].T
        y_hat = np.dot(weight.T,x_k)
        error = y[i]-y_hat
        #update weights
        weight_new = weight + 2*alpha*np.dot(x_k,error)
        #stroe weight
        w[i] = weight_new   

    return w

# compute NLMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha/ (sigma) *x[k]e[k]
def Normlized_LMS(x_stack, y, alpha, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """

    # compute nlms
    for i in range(1, max_iter):
        weight = w[i-1].T
        x_k = x_stack[i].T
        y_hat = np.dot(weight.T,x_k)
        error = y[i]-y_hat
        sigma = np.dot(x_k,x_k.T)/3 + 0.01
        #update weights
        weight_new = weight + 2*(alpha/sigma)*np.dot(x_k,error)
        #stroe weight
        w[i] = weight_new   

    return w

# compute RLS algorithm for N iterations  
# w[k+1] = np.dot(inv(R_x_hat), r_yx_hat)
def RLS(x_stack, y, gamma, delta_inv, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float gamma: forgetting factor
    :param float delta_inv: inverse of the auto-correlation matrix of x[k]
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """
    # parameter initialization
    R_x_inv = delta_inv * np.identity(3)
    r_yx_hat = np.zeros((3,1))

        # compute RLS
    for i in range(1, max_iter):
        
        x_k = x_stack[i].T        
        g = np.dot(R_x_inv, x_stack[i]) / (gamma**2 + np.dot(np.dot(x_k.T, R_x_inv), x_k))
        R_x_inv = gamma**(-2) * (R_x_inv - np.dot(np.dot(g, x_k.T), R_x_inv))
        r_yx_pre = np.array([np.dot(x_k.T, y[i])])
        r_yx_hat = np.multiply(gamma**2 , r_yx_hat)  + r_yx_pre.T
        
        #update weights
        weight_new = np.dot(R_x_inv,r_yx_hat)
       
        #stroe weight
        w[i] = weight_new.T   

    return w
    
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

    # set parameters for each algorithm
    alpha = 0.0001
    gamma = 1 - 1e-4
    delta_inv = 0.01
    
        
    """    # apply mean square error
    w_lms = Least_Mean_Square(x_stack, data_y, alpha, N, w)

    # apply normalized mean square error
    w_nlms = Normlized_LMS(x_stack, data_y, alpha, N, w) """

    # apply RLS
    w_RLS = RLS(x_stack, data_y, gamma, delta_inv, N, w)

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

    """     # plot the contour plot for LMS
    contour_plot(W0, W1, w_lms, J_vals, 
        title=f"LMS Contour Plot for alpha = {alpha}", 
        filename=f"LMS_contour_plot_alpha_{alpha}")

    # plot the contour plot for NLMS
    contour_plot(W0, W1, w_nlms, J_vals,
        title=f"NLMS Contour Plot for alpha = {alpha}",
        filename=f"NLMS_contour_plot_alpha_{alpha}")
    """
    # plot the contour plot for RLS
    contour_plot(W0, W1, w_RLS, J_vals,
        title=f"RLS Contour Plot for delta_inv = {delta_inv}",
        filename=f"RLS_contour_plot_delta_inv_{delta_inv}")
