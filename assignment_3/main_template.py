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

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('h0')
    ax.set_ylabel('h1')
    plt.show()

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

def test_model(model, test_loader, device):
    """ Test the trained model.
    Args:
        model (Model class): Trained model to test.
        x_test (torch.Tensor): Test data.
        y_test (torch.Tensor): Test labels.
    Returns:
        float: Accuracy of the model on the test data.
    """
    latent_list = []  # to store the latent representation of the whole dataset
    output_list = []  # to store the reconstructed output images of the whole dataset
    label_list = []  # to store the labels of the whole dataset

    test_loss = 0

    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):
                    
            # move to device
            x_clean = x_clean.to(device)

            # forward pass
            output, latent = model(x_clean)        
            latent = latent.detach().cpu()
            test_loss = criterion(output, x_clean)

            # append latent representation and the output of minibatch to the list
            latent_list.append(latent.detach().cpu())
            output_list.append(output.detach().cpu())
            label_list.append(test_label.detach().cpu())

            # add loss to the total loss
            test_loss += test_loss.item()

        # print the average loss for this epoch
        test_losses = test_loss / len(test_loader)

        print(f"Test loss is {test_losses}.")
        output = output.detach().cpu()
        
        return test_losses, output_list, latent_list, label_list



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
    AE = load_model(AE, "assignment_3/models/AE_model_best_50_epochs.pth")

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
   
    # get first minibatch
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    
    # excercise 1: get model output
    test_losses, output_list, latent_list, label_list = test_model(AE, test_loader, device)


    print("label list first 10 elements: ", labels_example)

    # concatenate all outputs into a tensor
    output_tensor = torch.cat(output_list, dim=0)
    latent_tensor = torch.cat(latent_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    print("Shape: {}".format(np.shape(output_tensor)))
    print("Shape: {}".format(np.shape(latent_tensor)))

    plot_images_exercise_1(x_clean_example, output_tensor[:10])

    # excercise 2: latent space
    scatter_plot(latent_tensor, label_tensor)