# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D://5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64

mu =
shrinkage = 
K = 

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]

# %% ISTA
def softthreshold(x,shrinkage):
    return

def ISTA(mu,shrinkage,K,y):
    return