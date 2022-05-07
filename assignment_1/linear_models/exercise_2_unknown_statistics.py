import numpy as np


# compute LMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*x[k]e[k]
def LMS(x_stack, y, alpha, max_iter, w_init):
    """
    :param np.array x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w_init(3, 1): initial weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T
    w = w_init

    # compute lms
    for i in range(max_iter - 1):

        # input/output data    
        x_k = x_stack[[i]].T
        y_k = y[i]

        # compute error
        y_hat = np.dot(w.T, x_k)
        error = y_k - y_hat

        # update weights
        w = w + 2 * alpha * x_k * error
        
        # store weights
        w_history[[i+1], :] = w.T 

    return w_history

# compute NLMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha/ (sigma) *x[k]e[k]
def NLMS(x_stack, y, alpha, max_iter, w_init):
    """
    :param np.array x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w_init(3, 1): initial weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T
    w = w_init

    # compute nlms
    for i in range(max_iter - 1):

        # input/output data    
        x_k = x_stack[[i]].T
        y_k = y[i]

        # compute error
        y_hat = np.dot(w.T, x_k)
        error = y_k - y_hat
        sigma = np.dot(x_k.T, x_k)/3 + 0.01

        # update weights
        w = w + 2*(alpha/sigma) * x_k * error
        
        # store weights
        w_history[[i+1], :] = w.T 

    return w_history

# compute RLS algorithm for N iterations  
# w[k+1] = np.dot(inv(R_x_hat), r_yx_hat)
def RLS(x_stack, y, gamma, delta_inv, max_iter, w_init):
    """
    :param np.array x_stack : x[k]
    :param np.array y : reference data_y
    :param float gamma: forgetting factor
    :param float delta_inv: inverse of the auto-correlation matrix of x[k]
    :param int max_iter: number of iterations
    :param np.array w_init(3, 1): initial weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T

    # initialize auto-correlation matrix
    R_x_inv = delta_inv * np.identity(3)
    r_yx = np.zeros((3, 1))

    # compute parameters
    for i in range(max_iter-1):

        # input/output data    
        x_k = x_stack[[i]].T
        y_k = y[i]

        # calculate parameters
        g = np.dot(R_x_inv, x_k) / (gamma**2 + np.dot(np.dot(x_k.T, R_x_inv), x_k))
        R_x_inv = gamma**(-2) * (R_x_inv - np.dot(np.dot(g, x_k.T), R_x_inv))
        r_yx = (gamma**2)*r_yx + x_k*y_k

        # update weights
        w = np.dot(R_x_inv, r_yx)        

        # store weights
        w_history[[i+1], :] = w.T

    return w_history
