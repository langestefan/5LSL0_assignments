import numpy as np


# compute SGD algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*(r_yx - R_x*w[k])
def SGD(R_x, r_yx, alpha, max_iter, w_init):
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
