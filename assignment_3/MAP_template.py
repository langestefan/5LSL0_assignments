# %% imports
# libraries
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

#import local files
import denoise_VAE
import MNIST_dataloader
import train_ex_7_to_8

def plot_MAP_loss(MAP_losses, save_path):
    """
    Plots the loss.
    -------
    train_losses: list
        The training loss
    valid_losses: list
        The validation loss
    """
    num_epochs = len(MAP_losses)

    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(MAP_losses, label='MAP loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('iterations[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    #plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 100))
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_images(x_noisy_test, map_out, x_clean_test):
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(3, 10, i+1)
        plt.imshow(x_noisy_test[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 10, i+10+1)
        plt.imshow(map_out[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 10, i+20+1)
        plt.imshow(x_clean_test[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'assignment_3/figures/exercise_8_b.png', dpi=300, bbox_inches='tight')
    plt.show()
# %% MAP Estimation
# parameters
data_loc = 'assignment_3/data' #change the data location to something that works for you
batch_size = 64
no_iterations = 1000
learning_rate = 1e-2
beta = 0.01

# get dataloader
train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
# take first 10 images from clean test data set
x_clean_test  = test_loader.dataset.Clean_Images[0:10]
# take first 10 images from noisy test data set
x_noisy_test  = test_loader.dataset.Noisy_Images[0:10]

# # use this for Exercise 8
model = denoise_VAE.VAE()

# # load the trained model
model = train_ex_7_to_8.load_model(model, "assignment_3/models/denoise_VAE_new_30_best.pth")

estimated_latent = nn.Parameter(torch.randn(10,16))
optimizer_map = torch.optim.Adam([estimated_latent],lr = learning_rate)

# optimization
MAP_losses = []
model.eval()
start_time = time.time()
for i in tqdm(range(no_iterations)):
    optimizer_map.zero_grad()
    output_decoder = model.decoder(estimated_latent)
    loss = ((x_noisy_test - output_decoder)**2).sum() + (beta*estimated_latent).sum()
    
    loss.backward()
    optimizer_map.step()
    MAP_losses.append(loss.item())
    
    #print(f'loss = {loss.item()}')
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
model.eval()
map_out = model.decoder(estimated_latent)

map_out = map_out.detach().numpy()
plot_images(x_noisy_test, map_out, x_clean_test)

plot_MAP_loss(MAP_losses, save_path='assignment_3/figures/excercise8b_MAP_loss.png')