# libraries
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

def scatter_plot(mnist_points):
    """
    Plot function from assignment document
    :param hw_points: Handwritten data feature vectors (digits, points, (x0, y0)) = (10, 1, ndim)
    :param mnist_points: MNIST feature vectors (digits, points, (x0, y0)) = (10, 20, ndim)
    """
    colors = plt.cm.Paired(np.linspace(0, 1, len(mnist_points)))
    fig, ax = plt.subplots(figsize=(7, 5))

    for (points, color, digit) in zip(mnist_points, colors, range(10)):
        ax.scatter([item[0] for item in points],
                   [item[1] for item in points],
                   color=color, label='digit{}'.format(digit))


    ax.grid(True)
    ax.legend()
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
   
    test_loss = 0
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):
        # fill in how to train your network using only the clean images

        # move to device
        x_clean = x_clean.to(device)

        # forward pass
        recon, latent = model(x_clean)
        
        latent = latent.detach().cpu()
        
    
        
        test_loss = criterion(recon, x_clean)
        # add loss to the total loss
        test_loss += test_loss.item()


    # print the average loss for this epoch
    test_losses = test_loss / len(test_loader)

    print(f"Test loss is {test_losses}.")
    #test_losses = test_losses.detech().cpu()
    recon = recon.detach().cpu()
    

    return test_losses,recon,latent,test_label



if __name__ == "__main__":

    # set torches random seed
    torch.random.manual_seed(0)

    # define parameters
    data_loc = 'data' #change the data location to something that works for you
    batch_size = 64
    n_epochs = 10
    learning_rate = 3e-4

    # get dataloader
    train_loader, valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # create the autoencoder
    AE = autoencoder_template.AE()

    # load the trained model 
    AE = load_model(AE, "AE_model_params.pth")

    # create the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)



    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    print("Using device:", device)

    # move model to device
    AE.to(device)

    # %% training loop
    #AE, train_losses, valid_losses = train.train_model (AE, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True)
    test_losses,test_recon, test_latent, test_label= test_model(AE, test_loader, device)
    
    scatter_plot(test_latent)
 


""" # # move back to cpu    
recon = recon.detach().cpu()
latent = latent.detach().cpu()
x_clean = x_clean.detach().cpu()
x_noisy = x_noisy.detach().cpu()

batch_size_TEST = 1

# get X_clean_exsample
x_clean_test  = test_loader.dataset.Clean_Images[0:10]

recon_test, latent_test = AE(x_clean_test.to(device))
recon_test = recon_test.detach().cpu()

#plot_images_exercise_1(x_clean_test, recon_test)
#plot_images_exercise_1(x_clean, recon)


#train.plot_loss(train_losses=train_losses, valid_losses=valid_losses) """

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