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
import autoencoder_template
import train

# nearest neighbor excercise 3
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

# to fix a bug with numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def scatter_plot(latent_tensor, label_tensor, n_points=10000):
    """
    Plot function from assignment document
    :param mnist_points: MNIST feature vectors (digits, points, (x0, y0)) = (10, 20, ndim)
    """
    
    colors = plt.cm.Paired(np.linspace(0, 1, 10)) # color map, 10 digits
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h'] # marker map, 10 digits
    fig, ax = plt.subplots(figsize=(10, 5))

    for (latent_xy, digit) in zip(latent_tensor[:n_points], label_tensor[:n_points]):
        color = colors[digit]
        ax.scatter([item[0] for item in latent_xy],
                   [item[1] for item in latent_xy],
                   color=color, s=20, label=f'digit{digit}', marker=markers[digit])

    ax.grid(True)

    # this trick makes sure all the labels in the legend are unique and only shown once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('h0')
    ax.set_ylabel('h1')
    plt.show()


def plot_images_exercise_1(x_data, model_output):

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

def nearest_neighbour_exercise_3(train_latent, test_latent, train_labels, test_labels):
    """
    Nearest neighbour exercise 3. See: https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    Args:
        train_latent: latent vectors of training data
        test_latent: latent vectors of test data
        train_labels: labels of training data
        test_labels: labels of test data
    """
    # create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_latent)
    digits = np.linspace(0, 9, 10, dtype=np.int8)

    # find the nearest neighbour for each image in x_data
    distances, train_latent_indices = nbrs.kneighbors(test_latent)

    # get the labels of the nearest neighbour
    train_labels_neighbours = train_labels[train_latent_indices]

    # compare the labels of the nearest neighbour and the test data
    correct_predictions = np.equal(train_labels_neighbours, test_labels)
    print("correct predictions shape: ", np.shape(correct_predictions))
    print("correct predictions ", correct_predictions)

    # calculate the accuracy by taking only the diagonal elements
    accuracy = np.mean(np.diagonal(correct_predictions))
    print("Accuracy: ", accuracy)

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(test_labels, train_labels_neighbours, normalize='true')
    print("Confusion matrix: \n", conf_matrix)
    print("Confusion matrix shape: ", np.shape(conf_matrix))

    # get percentages of each class
    class_percentages = np.diagonal(conf_matrix)
    print("Class percentages: ", class_percentages)

    # plot confusion matrix
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in digits],
                        columns = [i for i in digits])
    fig, ax = plt.subplots(figsize=(12, 7))
    heatmap = sn.heatmap(df_cm, annot=True)
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 16)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 16)

    # axis titles
    heatmap.set_ylabel('Groundtruth label', fontsize = 18)
    heatmap.set_xlabel('1-nearest neighbour classification', fontsize = 18)
    plt.savefig('assignment_3/figures/confusion_matrix_excercise_3.png', dpi=300, bbox_inches='tight')                       


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)

    # define parameters
    data_loc = 'data' #change the data location to something that works for you
    batch_size = 64
    n_epochs = 50
    learning_rate = 3e-4

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # create the autoencoder
    AE = autoencoder_template.AE()

    # load the trained model 
    AE = train.load_model(AE, "assignment_3/models/AE_model_best_50_epochs.pth")

    # create the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    AE.to(device)

    # train the model
    # AE, train_losses, valid_losses = train.train_model(AE, train_loader, 
    #                                                     valid_loader, optimizer, 
    #                                                     criterion, n_epochs, device, 
    #                                                     write_to_file=True)
   
    # get latent vectors for excercise 3, use the trained model on the train set
    n_epochs = 1
    losses_train, output_train, latent_train, label_train = train.test_model(AE, criterion, train_loader, device)

    # concatenate all train outputs into a tensor
    output_tensor_train = torch.cat(output_train, dim=0)
    latent_tensor_train = torch.cat(latent_train, dim=0)
    label_tensor_train = torch.cat(label_train, dim=0)
    print("Shape: {}".format(np.shape(output_tensor_train)))
    print("Shape: {}".format(np.shape(latent_tensor_train)))
    print("Shape: {}".format(np.shape(label_tensor_train)))
    
    # excercise 1: get model output
    losses_test, output_test, latent_test, label_test = train.test_model(AE, criterion, test_loader, device)

    # concatenate all test outputs into a tensor
    output_tensor_test = torch.cat(output_test, dim=0)
    latent_tensor_test = torch.cat(latent_test, dim=0)
    label_tensor_test = torch.cat(label_test, dim=0)
    print("Shape: {}".format(np.shape(output_tensor_test)))
    print("Shape: {}".format(np.shape(latent_tensor_test)))
    print("Shape: {}".format(np.shape(label_tensor_test)))

    # print the first 10 digits of test set (0-9)
    # examples = enumerate(test_loader)
    # _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # plot_images_exercise_1(x_clean_example, output_tensor_test[:10])


    ### excercise 2: latent space ###
    # scatter_plot(latent_tensor_test, label_tensor_test)

    ### excercise 3: 1-nearest neighbour classification ###
    latent_tensor_train = torch.squeeze(latent_tensor_train) # collapse 1-dim
    latent_tensor_test = torch.squeeze(latent_tensor_test) # collapse 1-dim
    print("latent_tensor_train: {}".format(np.shape(latent_tensor_train)))
    print("latent_tensor_test: {}".format(np.shape(latent_tensor_test)))

    nearest_neighbour_exercise_3(latent_tensor_train, latent_tensor_test, label_tensor_train, label_tensor_test)