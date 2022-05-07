import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift

# known/unknown statistics models
import exercise_1_known_statistics as known_stats
import exercise_2_unknown_statistics as unknown_stats

# contour plot function
def contour_plot(w0, w1, w_train, J_vals, title, filename, show=True):

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
    figure_name = f"assignment_1/linear_models/figures/{filename}.png"
    fig.savefig(figure_name, dpi=300)


def main():
    print(" ---- start ---- ")    
    # Load the data
    data = pd.read_csv("assignment_1/linear_models/assignment1_data.csv")

    # split data into x and y
    data_x = np.array(data.iloc[:, 0])
    data_y = np.array(data.iloc[:, 1])
    x_stack = np.vstack((data_x, shift(data_x, 1), shift(data_x, 2))).T

    # initialize the filter
    w_3_const = -0.5 # for countourplots
    w_init = np.zeros((3, 1))
    N = data.shape[0]


    # autocorrelation and cross-correlation matrices
    r_yx = np.array([[1, 5.3, -3.9]]).T 
    R_x = np.array([[5, -1, -2],
                    [-1, 5, -1],
                    [-2, -1, 5]])   

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

    # select the algorithm to use
    algorithm = "Newton"

    # show plots during runtime
    show = True

    # set parameters for each algorithm
    alpha = 0.001
    gamma = 1 - 1e-2
    delta_inv = 0.001

    ### known statistics, excercise 1 ###
    # apply steepest gradient descent
    if algorithm == "SGD":
        w_sgd = known_stats.SGD(R_x, r_yx, alpha, N, w_init)
        contour_plot(W0, W1, w_sgd, J_vals, 
            title=f"SGD Contour Plot for alpha = {alpha}", 
            filename=f"sgd_contour_plot_alpha_{alpha}",
            show=show)

    # apply newtons method
    elif algorithm == "Newton":
        w_newton = known_stats.newtons_method(R_x, r_yx, alpha, N, w_init)
        contour_plot(W0, W1, w_newton, J_vals,
            title=f"Newton's Method Contour Plot for alpha = {alpha}",
            filename=f"newtons_method_contour_plot_alpha_{alpha}",
            show=show)

    ### unknown statistics, excercise 2 ###
    # apply least mean squares
    elif algorithm == "LMS":
        w_lms = unknown_stats.LMS(x_stack, data_y, alpha, N, w_init)
        contour_plot(W0, W1, w_lms, J_vals, 
            title=f"LMS Contour Plot for alpha = {alpha}", 
            filename=f"LMS_contour_plot_alpha_{alpha}",
            show=show)       

    # apply normalized least mean squares
    elif algorithm == "NLMS":
        w_nlms = unknown_stats.NLMS(x_stack, data_y, alpha, N, w_init) 
        contour_plot(W0, W1, w_nlms, J_vals,
            title=f"NLMS Contour Plot for alpha = {alpha}",
            filename=f"NLMS_contour_plot_alpha_{alpha}",
            show=show)        
    
    # apply recursive least squares
    elif algorithm == "RLS":
        w_RLS = unknown_stats.RLS(x_stack, data_y, gamma, delta_inv, N, w_init)
        contour_plot(W0, W1, w_RLS, J_vals,
            title=f"RLS Contour Plot for gamma = {gamma}",
            filename=f"RLS_contour_plot_gamma_{gamma}",
            show=show)
    
    # algorithm not found
    else:
        print("Algorithm not found!")
        

if __name__ == "__main__":
    main()
