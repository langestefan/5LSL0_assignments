import torch

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


        # calculate validation loss
        valid_loss = calculate_loss(model, valid_loader, criterion, device)
        train_loss = calculate_loss(model, train_loader, criterion, device)

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
