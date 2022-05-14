# Q10: Given the solution presented in the lecture slides (2-13), reproduce the
# plots on slide 2-14 in e.g. Python, showing how the input is mapped into a the
# latent space h. Also plot the decision boundary f(x) = 0.5.

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return np.maximum(x, 0)

def main():
    # define X, y for XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    # define parameters
    W_1 = np.ones((2, 2))
    b_1 = np.array([[0, -1]]).T
    w_2 = np.array([[1, -2]]).T
    b_2 = np.array([[0]])

    # print all parameters
    print("W_1: \n", W_1)
    print("b_1: \n", b_1)
    print("w_2: \n", w_2)
    print("b_2: \n", b_2)
    print("-------------------------")

    h = np.zeros((4, 2), dtype=int)    

    # for every input x, calculate the output y
    for i, x in enumerate(X):
        print("x: \n", x)
        # calculate the output of the first layer
        h[[i]] = relu(np.dot(W_1, np.array([x]).T) + b_1).T
        print("h: \n", h[i])

    # plot x and h in separate plots
    plt.figure(figsize=(12, 5))
    max = 2.5
    min = -0.5

    # left plot: x
    ax = plt.subplot(1, 2, 1)
    plt.scatter(X[(0, 3), 0], X[(0, 3), 1], s=160, facecolors='none', edgecolors='r')
    plt.scatter(X[(1, 2), 0], X[(1, 2), 1], s=160, marker='x', c='b')
    plt.xlim(-0.5, max)
    plt.ylim(-0.5, 1.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')

    # right plot: h
    ax = plt.subplot(1, 2, 2)
    plt.scatter(h[(0, 3), 0], h[(0, 3), 1], s=160, facecolors='none', edgecolors='r')
    plt.scatter(h[(1, 2), 0], h[(1, 2), 1], s=160, marker='x', c='b')
    plt.xlim(-0.5, max)
    plt.ylim(-0.5, 1.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel('h1')
    plt.ylabel('h2')
    # decision boundary f(x) = 0.5
    # Define x and y values
    x = [-0.5, 3.5]
    y = [-0.5, 1.5]
    plt.plot(x, y, linewidth=2, color='green', label='f(x) = 0.5', linestyle='--')
    plt.legend()

    # save plot
    plt.savefig('assignment_2/Q10.png', dpi=300)




    plt.show()




if __name__ == '__main__':
    main()