# libraries
from turtle import shape
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

# from tqdm import tqdm
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import autoencoder_template
import train



def load_model(model, filename):
    """ Load the trained model.
    Args:
        model (Model class): Untrained model to load.
        filename (str): Name of the file to load the model from.
    Returns:
        Model: Model with parameters loaded from file.
    """
    model.load_state_dict(torch.load(filename))
    return model

def plot_images_exercise_1(x_data, recon):

    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_data[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2,10,i+11)
        plt.imshow(recon[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig("exercise_1.png",dpi=300,bbox_inches='tight')
    plt.show() 

# set torches random seed
torch.random.manual_seed(0)

# define parameters
data_loc = 'data' #change the data location to something that works for you
batch_size = 64
n_epochs = 30
learning_rate = 3e-4

# get dataloader
train_loader,valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
AE = autoencoder_template.AE()

# load the trained model 
#AE = load_model(AE, "AE_model_params.pth")

# create the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)
train_losses = []

# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu" # 
print("Using device:", device)

# move model to device
AE.to(device)

# %% training loop


# track losses
train_losses = []
valid_losses = []

# go over all epochs
for epoch in range(n_epochs):
    print(f"\nTraining Epoch {epoch}:")
    
    train_loss = 0

    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
        # fill in how to train your network using only the clean images

        # move to device
        x_clean = x_clean.to(device)
        x_noisy = x_noisy.to(device)
        label = label.to(device)

        # forward pass
        recon, latent = AE(x_clean)
        loss = criterion(recon, x_clean)
       

        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add loss to the total loss
        train_loss += loss.item()

    # calculate validation loss
    valid_loss = train.calculate_loss(AE, valid_loader, criterion, device)

    # print the average loss for this epoch
    train_loss /= len(train_loader)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"Average train loss for epoch {epoch} is {train_loss}, validation loss is {valid_loss}")

# write the model parameters to a file
torch.save(AE.state_dict(), "AE_model_params.pth")

# # move back to cpu    
recon = recon.detach().cpu()
latent = latent.detach().cpu()
x_clean = x_clean.detach().cpu()
x_noisy = x_noisy.detach().cpu()

batch_size_TEST = 1

# get X_clean_exsample
x_clean_test  = test_loader.dataset.Clean_Images[0:10]

recon_test, latent_test = AE(x_clean_test.to(device))
recon_test = recon_test.detach().cpu()

plot_images_exercise_1(x_clean_test, recon_test)
#plot_images_exercise_1(x_clean, recon)


train.plot_loss(train_losses=train_losses, valid_losses=valid_losses)

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