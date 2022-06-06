# %% imports
# libraries
from pyexpat.errors import XML_ERROR_XML_DECL
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# local imports
import MNIST_dataloader


# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
# x_clean_train = train_loader.dataset.Clean_Images
# x_noisy_train = train_loader.dataset.Noisy_Images
# labels_train  = train_loader.dataset.Labels

# x_clean_test  = test_loader.dataset.Clean_Images
# x_noisy_test  = test_loader.dataset.Noisy_Images
# labels_test   = test_loader.dataset.Labels

# # use these 10 examples as representations for all digits
# x_clean_example = x_clean_test[0:10,:,:,:]
# x_noisy_example = x_noisy_test[0:10,:,:,:]
# labels_example = labels_test[0:10]

# %% ISTA
def softthreshold(x,shrinkage):
    # x: torch.Size([1, 32, 32])
    x_thx = x
    
    # compare each pixels in x with shrinkage value
    for i in range(32):
        for j in range(32):
            if x[i,j] > shrinkage:
                x_thx[i,j] = ((np.abs(x[i,j]) - shrinkage)/np.abs(x[i,j]))*x[i,j]
            else:
                x_thx[i,j] = 0

    return x_thx

def ISTA(mu,shrinkage,K,y):

    x_new = y[0,:,:]
    x_old = y[0,:,:]
    A = np.identity(32)
    I = np.identity(32)

    for i in tqdm(range(K)):
        x_old = x_new        
        x_ista = mu*np.dot(A,x_old) + (I-mu*A*A.T)
        x_new = softthreshold(x_ista,shrinkage) 
    return x_new


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    
    # define parameters
    data_loc = 'assignment_4/data' #change the data location to something that works for you
    batch_size = 64
    mu = 1.4
    shrinkage = 0.1
    K = 20

    # get dataloader
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
    # take first 10 images from clean test data set: torch.Size([10, 1, 32, 32])
    x_clean_test  = test_loader.dataset.Clean_Images[0:10]
    # take first 10 images from noisy test data set: torch.Size([10, 1, 32, 32])
    x_noisy_test  = test_loader.dataset.Noisy_Images[0:10]
  
    #start timer
    start_time = time.time()

    # currently only take 1 image for ISTA
    # This is becasue I got problem for stack all the results together
    x_ista = ISTA(mu,shrinkage,K,x_noisy_test[10])
    plt.subplot(1,2,1)
    plt.imshow(x_ista,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(x_noisy_test[0,:,:],cmap='gray')
    plt.show()
  
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
 

   
   