# %% imports
# libraries
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

def plot_examples(clean_images, noisy_images, ista_output, num_examples=10):
    """
    Plots some examples from the dataloader.
    -------
    noisy_images: torch.Tensor
        The noisy images
    clean_images: torch.Tensor
        The clean images
    num_examples : int
        Number of examples to plot.
    """

    # show the examples in a plot
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(ista_output[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("assignment_4/figures/exercise_1_b.png", dpi=300, bbox_inches='tight')
    plt.show()

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

    A = np.identity(32)
    I = np.identity(32)

    for i in tqdm(range(K)):

        if i == 0:
            y = y
        else:
            y = x_out
        image_list = []
        for z in range(len(y)):            
            x_old = y[z,0,:,:]
            x_ista = mu*np.dot(A,x_old) + (I-mu*A*A.T)
            x_new = softthreshold(x_ista,shrinkage) 
            image_list.append(x_new)
           
        # convert to array
        x_out = np.array(image_list)
        
        # convert to tensor
        x_out = torch.from_numpy(x_out).float()
        x_out = x_out.unsqueeze(1)

        #print("x_out:",np.shape(x_out))


    return x_out


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    
    # define parameters
    data_loc = 'assignment_4/data' #change the data location to something that works for you
    batch_size = 64
    mu = 1.5
    shrinkage = 0.1
    K = 50

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
    
    x_ista = ISTA(mu,shrinkage,K,x_noisy_test)

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    plot_examples(x_clean_test, x_noisy_test, x_ista)
 

   
   