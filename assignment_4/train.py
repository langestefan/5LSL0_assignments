import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm as tqdm

def reshape_images(images):
    """
    Reshapes the images to be 32x32x1
    -------
    images: torch.Tensor
        The images to reshape
    """
    # reshape the images to be 32x32x1
    images_reshaped = torch.reshape(images, (images.shape[0], 1, 32, 32))
    return images_reshaped


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


def plot_examples(clean_images, noisy_images, ista_output, num_examples=10):
    """
    Plots some examples from the dataloader.
    -------
    noisy_images: torch.Tensor
        The noisy images
    clean_images: torch.Tensor
        The clean images
    num_examples : int
        Number of examples to plot.
    """

    print("input size: ", noisy_images.shape)
    print("output size: ", ista_output.shape)
    print("clean_images size: ", clean_images.shape)


    # show the examples in a plot
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.imshow(noisy_images[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(ista_output[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(clean_images[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    #plt.savefig("assignment_4/figures/exercise_1_b.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss(train_losses, valid_losses, save_path):
    """
    Plots the loss.
    -------
    train_losses: list
        The training loss
    valid_losses: list
        The validation loss
    """
    num_epochs = len(train_losses)

    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_losses, label='Training loss')
    ax.plot(valid_losses, label='Validation loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()


# calculate validation loss
def calculate_loss(model, data_loader, criterion, device):
    """
    Calculate the loss on the given data set.
    -------
    model : model class
        Model structure to fit, as defined by build_model().
    data_loader : torch.utils.data.DataLoader
        Data loader to use for the data set.
    criterion : torch.nn.modules.loss
        Loss function to use.
    device : torch.device
        Device to use for the model.
    -------
    loss : float    
        The loss on the data set.
    """
    # set model to evaluation mode
    model.eval()

    # initialize loss
    loss = 0

    # loop over batches
    for batch_idx, (x_clean, x_noisy, label) in enumerate(tqdm(data_loader, position=0, leave=False, ascii=False)):

            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                x_clean, x_noisy = [x.cuda() for x in [x_clean, x_noisy]]
                model.to(device)

            # forward pass
            x_out = model(x_noisy)
           
            # calculate loss
            loss += criterion(x_out, x_clean).item()

    # return the loss
    return loss / len(data_loader)
    # return loss

def test_ex2c(model, criterion, test_loader):

    model.eval()
    LISTA_mse_losses = 0
    loss = 0
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
        
        x_ista = model(x_noisy)
        loss = criterion(x_ista,x_clean)
        LISTA_mse_losses += loss.item()
   
    print(f'test_loss = {LISTA_mse_losses/len(test_loader)}') 



# train model function
def train_model(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True, save_path=None):
    """
    Fit the model on the training data set.
    Arguments
    ---------
    model : model class
        Model structure to fit, as defined by build_model().
    train_loader : torch.utils.data.DataLoader
        Dataloader for the training set.
    valid_loader : torch.utils.data.DataLoader
        Dataloader for the validation set.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training.
    criterion : torch.nn.modules.loss
        Loss function to use for training.
    epochs : int
        Number of epochs to train for.
    device : torch.device
        Device to use for training.
    write_to_file : bool
        Whether to write the model parameters to a file.
    path_to_save_model : str
        Path to save the model parameters to.

    Returns
    -------
    model : model class
        The trained model.
    training_losses : list
        The training loss for each epoch.
    validation_losses : list
        The validation loss for each epoch.
    """
    # to keep track of loss
    train_losses = []
    valid_losses = []

    # go over all epochs
    for epoch in range(n_epochs):
        print(f"\nTraining Epoch {epoch}:")
        
        train_loss = 0
        valid_loss = 0

        # go over all minibatches
        for batch_idx, (x_clean, x_noisy, __) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
            # move to device
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            model = model.to(device)

            # forward pass
            x_out = model(x_noisy)
            loss = criterion(x_out, x_clean) 
        
            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()

        # calculate validation loss
        # valid_loss = calculate_loss_ex4(model, valid_loader, criterion, device) # classifier
        valid_loss = calculate_loss(model, valid_loader, criterion, device) # autoencoder 

        # average loss for this epoch = train_loss / n_batches
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Average train loss for epoch {epoch} is {train_loss}, validation loss is {valid_loss}")

        # write the model parameters to a file every 5 epochs
        if write_to_file and epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_{epoch}_epochs.pth")

    if write_to_file:
        torch.save(model.state_dict(), f"{save_path}_{epoch}_epochs.pth")

    # return the trained model
    return model, train_losses, valid_losses
