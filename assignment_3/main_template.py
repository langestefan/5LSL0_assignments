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

# create the autoencoder
AE = autoencoder_template.AE()

# create the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)
train_losses = []

# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# move model to device
AE.to(device)

# %% training loop
# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    
    train_loss = 0

    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images

        # forward pass
        recon, latent = AE(x_clean)
        loss = criterion(recon, x_clean)
        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add loss to the total loss
        train_loss += loss.item()

# show the examples in a plot
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_clean[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,10,i+11)
    plt.imshow(recon[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("autoencoder_output.png",dpi=300,bbox_inches='tight')
plt.show() 

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
""" 
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels """

# use these 10 examples as representations for all digits
""" 
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10] """