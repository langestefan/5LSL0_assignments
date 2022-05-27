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
    n_epochs = 1
    learning_rate = 3e-4
    reconstruction_term_weight = 1

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # # create the autoencoder
    model = VAE.VAE()

    # # load the trained model 
    # model = train.load_model(model, "assignment_3/models/excercise1/AE_model_best_50_epochs.pth")

    # create the optimizer
    criterion_ex1 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    model.to(device)

    # train the model excercise 1
    
    model, train_kl_losses, train_reconstruction_losses, train_batch_losses, train_epoch_losses = train_ex_7_to_8.train_model(reconstruction_term_weight, model, train_loader, 
                                                                                                                                valid_loader, optimizer, 
                                                                                                                                criterion_ex1, n_epochs, device, 
                                                                                                                                write_to_file=True,
                                                                                                                                save_path='assignment_3/models/VAE')
    

    # get latent vectors for excercise 3, use the trained model on the train set
    # losses_train, output_train, latent_train, label_train = train.test_model(AE, criterion_ex1, train_loader, device)

    # # concatenate all train outputs into a tensor
    # output_tensor_train = torch.cat(output_train, dim=0)
    # latent_tensor_train = torch.cat(latent_train, dim=0)
    # label_tensor_train = torch.cat(label_train, dim=0)
    
    # # excercise 1: get model output
    # losses_test, output_test, latent_test, label_test = train.test_model(AE, criterion_ex1, test_loader, device)

    # # concatenate all test outputs into a tensor
    # output_tensor_test = torch.cat(output_test, dim=0)
    # latent_tensor_test = torch.cat(latent_test, dim=0)
    # label_tensor_test = torch.cat(label_test, dim=0)

    # print the first 10 digits of test set (0-9)
    # examples = enumerate(test_loader)
    # _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # plot_images_exercise_1(x_clean_example, output_tensor_test[:10])
