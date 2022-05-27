# libraries
from matplotlib import markers
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


# from tqdm import tqdm
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import train_ex_7_to_8
import VAE

# nearest neighbor excercise 3
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#import seaborn as sn

# to fix a bug with numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    #plt.savefig("exercise_1.png",dpi=300,bbox_inches='tight')
    plt.show() 

if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    # define parameters
    data_loc = 'assignment_3/data' #change the data location to something that works for you
    batch_size = 64
    n_epochs = 5
    learning_rate = 3e-4

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # # create the autoencoder
    model = VAE.VAE()

    # # load the trained model 
    #model = train_ex_7_to_8.load_model(model, "assignment_3/models/VAE_5_epochs.pth")

    # create the optimizer
    criterion_ex1 = F.mse_loss
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    model.to(device)

    # train the model excercise 1
    
    model, train_kl_losses, train_reconstruction_losses, train_epoch_losses = train_ex_7_to_8.train_model(model, train_loader, 
                                                                                                    valid_loader, optimizer, 
                                                                                                    criterion_ex1, n_epochs, device, 
                                                                                                    write_to_file=True,
                                                                                                    save_path='assignment_3/models/VAE')
    #print("kl_losses: ",  train_kl_losses)
    

    # # excercise 7a: get model output
    # test_losses, output_list, latent_list, label_list = train_ex_7_to_8.test_model(model, criterion_ex1, test_loader, device)
    # print("shape of output_list: ", [t.size() for t in output_list])

    #plot the loss
    train_ex_7_to_8.plot_loss(train_kl_losses, train_reconstruction_losses, save_path='assignment_3/figures/excercise7a_loss.png')

    # # concatenate all test outputs into a tensor
    # output_tensor_test = torch.cat(output_list, dim=0)
    # latent_tensor_test = torch.cat(latent_test, dim=0)
    # label_tensor_test = torch.cat(label_test, dim=0)

    # # print the first 10 digits of test set (0-9)
    # examples = enumerate(test_loader)
    # _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # plot_images_exercise_7a(x_clean_example, output_tensor_test[:10])
