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


def plot_examples(clean_images, noisy_images, prediction, num_examples=10):
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

    # show the examples in a plot
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(prediction[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("data_examples.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss(train_losses, valid_losses):
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
    plt.xticks(np.arange(0, num_epochs, 10))
    plt.savefig("loss.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_examples(noisy_images, clean_images, num_examples=10):
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

    # show the examples in a plot
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        plt.subplot(2, num_examples, i+1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2, num_examples, i + num_examples + 1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("data_examples.png", dpi=300, bbox_inches='tight')
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
    for batch_idx, (clean_images, noisy_images, labels) in enumerate(tqdm(data_loader, position=0, leave=False, ascii=False)):

            # flatten the images
            #clean_images = clean_images.view(clean_images.shape[0], -1)
            #noisy_images = noisy_images.view(noisy_images.shape[0], -1)   

            # move to GPU if available
            #noisy_images = noisy_images.to(device),
            clean_images = clean_images.to(device)

            # forward pass
            outputs, latent = model(clean_images)
           
            # calculate loss
            loss += criterion(outputs, clean_images).item()

    # return the loss
    return loss / len(data_loader)
    # return loss
            

# train model function
def train_model(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True):
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

    Returns
    -------
    model : model class
        The trained model.
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
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
            # fill in how to train your network using only the clean images

            # move to device
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            label = label.to(device)

            # forward pass
            output, latent = model(x_clean)
            loss = criterion(output, x_clean)
        
            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()

        # calculate validation loss
        valid_loss = calculate_loss(model, valid_loader, criterion, device)

        # print the average loss for this epoch
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Average train loss for epoch {epoch} is {train_loss}, validation loss is {valid_loss}")

        # write the model parameters to a file every 5 epochs
        if write_to_file and epoch % 5 == 0:
            torch.save(model.state_dict(), f"assignment_3/models/AE_model_checkpoint_{epoch+1}_epochs.pth")

    if write_to_file:
        torch.save(model.state_dict(), f"assignment_3/models/AE_model_best_{epoch+1}_epochs.pth")

    # return the trained model
    return model, train_losses, valid_losses
