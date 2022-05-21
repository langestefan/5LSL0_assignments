# %% imports
# libraries
from turtle import shape
import torch.optim as optim
import torch.nn as nn
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
data_loc = 'data' #change the data location to something that works for you
batch_size = 64
no_epochs = 1
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

""" 
for idx,x_clean_train in enumerate (train_loader.dataset.Clean_Images[10:20]):

    clean_images = x_clean_train.view(x_clean_train.shape[0], -1)
    print ("Clean images shape:", clean_images.shape) """



# create the autoencoder
AE = autoencoder_template.AE()

# create the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)
train_losses = []


# %% training loop
# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    
    train_loss = 0

    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images

        # forward pass
        recon,latent = AE(x_clean)
        print ("Latent shape:", latent.shape)
        print ("Recon shape:", recon.shape)
        print ("Clean images shape:", x_clean.shape)
        loss = criterion(recon, x_clean)
        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add loss to the total loss
        train_loss += loss.item()


# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
""" x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels """

# use these 10 examples as representations for all digits
""" x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10] """