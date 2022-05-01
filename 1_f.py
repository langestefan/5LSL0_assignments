import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error

# load data
data = pd.read_csv("assignment1_data.csv", header=None)

#split data into x and y
data_x = np.array(data.iloc[:, 0])
data_y = np.array(data.iloc[:, 1])
#print(raw_x.shape,data_x.shape)'

# make Rx and ryx matrix
R_x = np.array([[5., -1., -2.], [-1., 5., -1.], [-2., -1., 5.]])
r_yx = np.array([1, 5.3, -3.9]).T

#prepare parameters for later use
iterations = 5000
x_stack = np.vstack((data_x, shift(data_x,1), shift(data_x,2))).T
w =np.zeros((iterations,3))
J = np.zeros((iterations,1))
alpha=0.001

# weights updates
for i in range (1,iterations):
    weight = w[i-1].T
    weight_new = weight + 2*alpha*(r_yx-np.dot(R_x,weight))
    J[i] = mean_squared_error(data_y,np.dot(x_stack,weight_new))
    w[i] = weight_new
    
# make coutour plot    
w0 = np.linspace(-0.5, 0.5, 100)
w1 = np.linspace(-0.1, 1.5, 100)
W0,W1 = np.meshgrid(w0, w1)
J_vals = np.zeros((len(W0),len(W1)))

for i in range(len(W0)):
    for j in range(len(W1)):
        w_temp = np.array([W0[0,i], W1[j,0], -0.5]).T
        J_vals[i,j] =  mean_squared_error(data_y,np.dot(x_stack,w_temp))
                                           
fig = plt.figure()
cp = plt.contour(w0, w1, J_vals)
plt.clabel(cp, inline=1, fontsize=10)
plt.plot(w[:,0], w[:,1])
plt.title('SGD Contours Plot with adaption rate = 0.001')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()
