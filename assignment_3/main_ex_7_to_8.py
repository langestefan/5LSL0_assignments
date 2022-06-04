# libraries

from matplotlib import markers
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


# from tqdm import tqdm
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import train_ex_7_to_8
import VAE
import denoise_VAE

# nearest neighbor excercise 3
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#import seaborn as sn

# to fix a bug with numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_images_exercise_8(model, test_loader):

    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)

    output_decoder, x_sample, x_mean, x_log_var  = model(x_clean_example.cuda())
    output_decoder = output_decoder.data.cpu().numpy()

    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(3, 10, i+1)
        plt.imshow(x_noisy_example[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 10, i+10+1)
        plt.imshow(output_decoder[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 10, i+20+1)
        plt.imshow(x_clean_example[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'assignment_3/figures/exercise_8_b.png', dpi=300, bbox_inches='tight')
    plt.show()


def scatter_plot(latent_tensor, label_tensor, n_points=10000):
    """
    Plot function from assignment document
    :param mnist_points: MNIST feature vectors (digits, points, (x0, y0)) = (10, 20, ndim)
    """

    colors = plt.cm.Paired(np.linspace(0, 1, 10)) # color map, 10 digits
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h'] # marker map, 10 digits
    fig, ax = plt.subplots(figsize=(10, 5))

    for (latent_xy, digit) in zip(latent_tensor[:n_points], label_tensor[:n_points]):
        print("latent xy:",latent_xy)
        print("digit:",digit)
        color = colors[digit]
        ax.scatter( latent_xy[0],latent_xy[1],color=color, s=20, label=f'digit{digit}', marker=markers[digit])

    ax.grid(True)

    # this trick makes sure all the labels in the legend are unique and only shown once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('h0')
    ax.set_ylabel('h1')
    plt.show()


def plot_images_exercise_7a(x_data, model_output):

    print("shape output: ", np.shape(model_output))

    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_data[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,10,i+11)
        plt.imshow(model_output[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    # plt.savefig("exercise_1.png",dpi=300,bbox_inches='tight')
    plt.show()

def plot_mnist_grid_excercise_7d(images, n_img_x, n_img_y, save_path=None):
    """
    Plot MNIST images in a grid
    Args:
        images: MNIST images (N, 1, 32, 32)
    """
    # plot the images in a grid
    plt.figure(figsize=(12, 12))
    for j in range(n_img_y):
        for i in range(n_img_x):
            img_idx = i + j * n_img_x + 1
            plt.subplot(n_img_x, n_img_y, img_idx)
            plt.imshow(images[img_idx-1, 0, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'{save_path}/exercise_7d_1.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    # define parameters
    data_loc = 'data' #change the data location to something that works for you
    batch_size = 64
    n_epochs = 35
    learning_rate = 1e-3

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # # create the autoencoder
    model = VAE.VAE()
    # # use this for Exercise 8
    #model = denoise_VAE.VAE()

    # load the trained model
    model = train_ex_7_to_8.load_model(model, "assignment_3/models/VAE_35_best.pth")


    # create the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    model.to(device)

    # train the model excercise 7
    # model, x_sample, output_decoder, train_loss, test_loss = train_ex_7_to_8.train(model,
    #                                                                                 optimizer,
    #                                                                                 n_epochs,
    #                                                                                 train_loader,
    #                                                                                 test_loader,
    #                                                                                 save_path='assignment_3/models/VAE')

    # excercise 7a: get model output
    test_losses, output_list, latent_test, label_test = train_ex_7_to_8.test_model(model, criterion, test_loader, device)


    # plot the loss
    # train_ex_7_to_8.plot_loss(train_loss, test_loss, save_path='assignment_3/figures/excercise7a_smaller_loss.png')
    # train_ex_7_to_8.plot_kl_loss(train_kl_losses, save_path='assignment_3/figures/excercise7a_kl_loss.png')


    # # concatenate all test outputs into a tensor
    output_tensor_test = torch.cat(output_list, dim=0)
    latent_tensor_test = torch.cat(latent_test, dim=0)
    label_tensor_test = torch.cat(label_test, dim=0)
    # print("shape output_tensor_test: ", np.shape(output_tensor_test))
    # print("shape latent_tensor_test: ", np.shape(latent_tensor_test))
    # print("shape label_tensor_test: ", np.shape(label_tensor_test))
    # output_tensor_test = np.shape(output_tensor_test)
    # latent_tensor_test = np.array(latent_tensor_test)
    # label_tensor_test = np.shape(label_tensor_test)

    # print the first 10 digits of test set (0-9)
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    plot_images_exercise_7a(x_clean_example, output_tensor_test[:10])

    ## excercise 7b: latent space ###
    scatter_plot(latent_tensor_test, label_tensor_test)

    # excercise 7d: Decoder sample generation for VAE ###
    x_coords = np.linspace(-3, 3, 15)
    y_coords = np.linspace(-3, 3, 15)[::-1] # the [::-1] reverses the array so that the y-axis is flipped
    h0, h1 = np.meshgrid(x_coords, y_coords)

    # create a grid of latent vectors. Each row is a latent vector (h0, h1) point in the (15, 15) grid
    latent_grid = np.stack((h0.flatten(), h1.flatten()), axis=1)
    #print(np.shape(latent_grid))

    # plt.figure(figsize=(12,6))
    # plt.plot(latent_grid[:, 0], latent_grid[:, 1], marker='.', color='k', linestyle='none')
    # plt.show()

    # create tensor of size (N,1,2,1) where N = batchsize
    latent_grid_tensor = torch.from_numpy(latent_grid).float().to(device)
    #print(np.shape(latent_grid_tensor))

    # # get the decoder output
    # model.eval()
    # decoder_output = model.decoder(latent_grid_tensor).detach().cpu().numpy()
    # #print(np.shape(decoder_output))

    # plot_mnist_grid_excercise_7d(decoder_output, n_img_x=15, n_img_y=15,
    #                             save_path='assignment_3/figures')

    ### excercise 8a: Noisy image input to Variational auto-encoder ###
    #model, x_sample, output_decoder, train_loss, test_loss = train_ex_7_to_8.train(model,optimizer,30,train_loader,test_loader,save_path='assignment_3/models/denoise_VAE')

    # plot the first 10 digits of test set (0-9)
    #plot_images_exercise_8(model,test_loader)