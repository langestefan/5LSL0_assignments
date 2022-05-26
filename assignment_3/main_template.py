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

def plot_confusion_matrix(true_labels, output_labels, save_path=None):
    """
    Plot confusion matrix. See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Args:
        true_labels: Groud truth labels
        output_labels: Model output labels
        title: title of the plot
        cmap: color map
    """
    digits = np.linspace(0, 9, 10, dtype=np.int8)

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, output_labels, normalize='true')

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
    heatmap.set_xlabel('Model predicted label', fontsize = 18)
    plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')   

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
    

    # find the nearest neighbour for each image in x_data
    distances, train_latent_indices = nbrs.kneighbors(test_latent)

    # get the labels of the nearest neighbour
    train_labels_neighbours = train_labels[train_latent_indices]

    # compare the labels of the nearest neighbour and the test data
    # calculate the accuracy by taking only the diagonal elements
    correct_predictions = np.equal(train_labels_neighbours, test_labels)
    accuracy = np.mean(np.diagonal(correct_predictions))
    print("Accuracy: ", accuracy)

    # plot confusion matrix
    plot_confusion_matrix(test_labels, train_labels_neighbours, save_path='assignment_3/figures/confusion_matrix_ex3.png')



def plot_mnist_grid_excercise_5(images, n_img_x, n_img_y, save_path=None):
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
    plt.savefig(f'{save_path}/exercise_5.png', dpi=300, bbox_inches='tight')
    plt.show()



def plot_images_exercise_6(noisy_images, model_out_images, clean_images, save_path=None):
    """
    plot conisting of 10 collumns and 3 rows. Each collumn should show one of the digits. 
    Row 1 should show the noisy input, row 2 should show the output, 
    row 3 should showthe corresponding clean image x_clean_example.

    Args:
        noisy_images: noisy images
        model_out_images: model output images
        clean_images: clean images
    """
    # plot the images in a grid
    plt.figure(figsize=(12, 4))
    for j in range(3):
        for i in range(10):
            img_idx = i + j * 10 + 1
            digit_idx = i

            plt.subplot(3, 10, i+1)
            plt.imshow(noisy_images[digit_idx, 0, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 10, i+10+1)
            plt.imshow(model_out_images[digit_idx, 0, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 10, i+20+1)
            plt.imshow(clean_images[digit_idx, 0, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'{save_path}/exercise_6.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)

    # define parameters
    data_loc = 'data' #change the data location to something that works for you
    batch_size = 64
    # n_epochs = 50
    # learning_rate = 3e-4

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # # create the autoencoder
    AE = autoencoder_template.AE()

    # # load the trained model 
    AE = train.load_model(AE, "assignment_3/models/excercise1/AE_model_best_50_epochs.pth")

    # create the optimizer
    criterion_ex1 = nn.MSELoss()
    # optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    AE.to(device)

    # train the model excercise 1
    # AE, train_losses, valid_losses = train.train_model(AE, train_loader, 
    #                                                     valid_loader, optimizer, 
    #                                                     criterion_ex1, n_epochs, device, 
    #                                                     write_to_file=True)

    # get latent vectors for excercise 3, use the trained model on the train set
    # losses_train, output_train, latent_train, label_train = train.test_model(model, criterion_ex1, train_loader, device)

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


    ### excercise 2: latent space ###
    # scatter_plot(latent_tensor_test, label_tensor_test)

    ### excercise 3: 1-nearest neighbour classification ###
    # latent_tensor_train = torch.squeeze(latent_tensor_train) # collapse 1-dim
    # latent_tensor_test = torch.squeeze(latent_tensor_test) # collapse 1-dim

    # nearest_neighbour_exercise_3(latent_tensor_train, latent_tensor_test, label_tensor_train, label_tensor_test)


    ### excercise 4: Classifier network ###
    # n_epochs = 20
    # learning_rate = 0.0001

    # criterion_ex4 = nn.CrossEntropyLoss()
    # mnist_classifier = autoencoder_template.Classifier()
    # mnist_classifier.to(device)
    # optimizer_ex4 = optim.Adam(mnist_classifier.parameters(), learning_rate, weight_decay=1e-7)

    # AE, train_losses, valid_losses = train.train_model(mnist_classifier, train_loader=train_loader, 
    #                                                     valid_loader=valid_loader, optimizer=optimizer_ex4, 
    #                                                     criterion=criterion_ex4, n_epochs=n_epochs, 
    #                                                     device=device, 
    #                                                     write_to_file=True, 
    #                                                     path_to_save_model='assignment_3/models/excercise4/classifier')


    # plot the loss
    # train.plot_loss(train_losses, valid_losses, save_path='assignment_3/figures/excercise4_loss.png')

    # load pretrained model
    # mnist_classifier = train.load_model(mnist_classifier, 'assignment_3/models/excercise4/classifier_19_epochs.pth')

    # test_losses, pred_label_list, gt_label_list = train.test_model_ex4(mnist_classifier, 
    #                                                                    criterion_ex4, 
    #                                                                    test_loader, device)

    # # concatenate all train outputs into a tensor
    # pred_tensor_test = torch.cat(pred_label_list, dim=0)    
    # gt_tensor_test = torch.cat(gt_label_list, dim=0)

    # print("pred_tensor_test size: ", np.shape(pred_tensor_test))
    # print("gt_tensor_test size: ", np.shape(gt_tensor_test))

    # # plot confusion matrix
    # plot_confusion_matrix(gt_tensor_test, pred_tensor_test, save_path='assignment_3/figures/confusion_matrix_class_ex4.png')


    ### excercise 5: Decoder sample generation ###
    # x_coords = np.linspace(0.3, 3, 15)
    # y_coords = np.linspace(0.2, 5.5, 15)[::-1] # the [::-1] reverses the array so that the y-axis is flipped
    # h0, h1 = np.meshgrid(x_coords, y_coords)

    # # create a grid of latent vectors. Each row is a latent vector (h0, h1) point in the (15, 15) grid
    # latent_grid = np.stack((h0.flatten(), h1.flatten()), axis=1)
    # # print(latent_grid)

    # # plt.figure(figsize=(12,6))
    # # plt.plot(latent_grid[:, 0], latent_grid[:, 1], marker='.', color='k', linestyle='none')
    # # plt.show()

    # # create tensor of size (N,1,2,1) where N = batchsize
    # latent_grid_tensor = torch.from_numpy(latent_grid).float().to(device)
    # latent_grid_tensor = latent_grid_tensor.unsqueeze(1).unsqueeze(3)
    # print(np.shape(latent_grid_tensor))

    # # get the decoder output
    # AE.eval()
    # decoder_output = AE.decoder(latent_grid_tensor).detach().cpu().numpy()
    # print(np.shape(decoder_output))

    # plot_mnist_grid_excercise_5(decoder_output, n_img_x=15, n_img_y=15,
    #                             save_path='assignment_3/figures')

    
    ### excercise 6: Noisy image input to auto-encoder ###
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
        
    # get model output from noisy input images
    __, output_test, __, label_test = train.test_model(AE, criterion_ex1, test_loader, device, use_noisy_images=True)

    # concatenate all test outputs into a tensor
    output_tensor_test = torch.cat(output_test, dim=0)
    label_tensor_test = torch.cat(label_test, dim=0)

    plt.figure(figsize=(12,6))
    plt.imshow(output_tensor_test[0, 0, :, :], cmap='gray')
    plt.show()

    # plot the first 10 digits of test set (0-9)    
    plot_images_exercise_6(x_noisy_example[:10], output_tensor_test[:10], 
                           x_clean_example[:10], save_path='assignment_3/figures/')

