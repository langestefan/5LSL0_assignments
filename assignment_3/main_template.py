# %% imports
# libraries
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import autoencoder_template

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D://5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 4
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
AE = autoencoder_template.AE()

# create the optimizer


# %% training loop
# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images
        1+1


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