import torch

import matplotlib.pyplot as plt
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
    for batch_idx, (clean_images, noisy_images, labels) in enumerate(data_loader):

            # flatten the images
            clean_images = clean_images.view(clean_images.shape[0], -1)
            noisy_images = noisy_images.view(noisy_images.shape[0], -1)   

            # move to GPU if available
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            # forward pass
            outputs = model(noisy_images)

            # calculate loss
            loss += criterion(outputs, clean_images).item()

    # return the loss
    return loss / len(data_loader)
    # return loss
            

# train model function
def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs, device, write_to_file=False):
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

    # loop over epochs
    for epoch in range(epochs):

        # set model to train mode
        model.train()

        # initialize loss
        train_loss = 0

        # loop over batches
        for batch_idx, (clean_images, noisy_images, labels) in enumerate(train_loader):

            # flatten the images
            clean_images = clean_images.view(clean_images.shape[0], -1)
            noisy_images = noisy_images.view(noisy_images.shape[0], -1)                   

            # print("Clean images shape:", clean_images.shape)
            # print("Noisy images shape:", noisy_images.shape)
            
            # move to GPU if available
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            # clear the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(noisy_images)               
            loss = criterion(outputs, clean_images)

            # backwards pass, update weights
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()

            # print training loss
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(clean_images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # calculate training loss
        train_loss /= len(train_loader)

        # calculate validation loss
        valid_loss = calculate_loss(model, valid_loader, criterion, device)

        print("Epoch: {}/{} ".format(epoch+1, epochs),
                "Training Loss: {:.3f} ".format(train_loss),
                "Validation Loss: {:.3f}".format(valid_loss))

        # append losses
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    # write the model parameters to a file
    if write_to_file:
        torch.save(model.state_dict(), "model_params.pth")


    # return the trained model
    return model, train_losses, valid_losses
