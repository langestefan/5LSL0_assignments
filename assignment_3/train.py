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


def test_model(model, criterion, test_loader, device, use_noisy_images=False):
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
    label_list = []  # to store the groundtruth labels of the whole dataset

    test_loss = 0

    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):

            image_batch = x_clean if use_noisy_images else x_noisy
            x_clean = x_clean.to(device)

            # forward pass
            image_batch = image_batch.to(device)
            output, latent = model(image_batch)
            
            output = output.to(device)            
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


# test model excercise 4
def test_model_ex4(model, criterion, test_loader, device):
    """ Test the trained model.
    Args:
        model (Model class): Trained model to test.
        x_test (torch.Tensor): Test data.
        y_test (torch.Tensor): Test labels.
    Returns:
        float: Accuracy of the model on the test data.
    """
    pred_label_list = []  # to store the predicted labels
    gt_label_list = []  # to store the groundtruth labels 

    test_loss = 0

    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):
                    
            # move to device
            x_clean = x_clean.to(device)

            # forward pass
            pred_labels = model(x_clean)  
            test_loss = criterion(pred_labels, test_label.to(device))

            # convert pred_labels probabilities to class labels
            # indice position also corresponds to the class label. Pos 0 = digit 0, pos 1 = digit 1, pos 2 = digit 2, etc.
            pred_labels = torch.argmax(pred_labels, dim=1)

            # append outputs of minibatch to the list
            pred_label_list.append(pred_labels.detach().cpu())
            gt_label_list.append(test_label.detach().cpu())

            # add loss to the total loss
            test_loss += test_loss.item()

        # print the average loss for this epoch
        test_losses = test_loss / len(test_loader)

        print(f"Test loss is {test_losses}.")
        
        return test_losses, pred_label_list, gt_label_list



# calculate validation loss for excercise 4 (classification)
def calculate_loss_ex4(model, data_loader, criterion, device):
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
    for batch_idx, (clean_images, noisy_images, target_labels) in enumerate(tqdm(data_loader, position=0, leave=False, ascii=False)):

            # flatten the images
            #clean_images = clean_images.view(clean_images.shape[0], -1)
            #noisy_images = noisy_images.view(noisy_images.shape[0], -1)   

            # move to GPU if available
            #noisy_images = noisy_images.to(device),
            clean_images = clean_images.to(device)

            # forward pass
            output_labels = model(clean_images)
           
            # calculate loss
            loss += criterion(output_labels, target_labels.to(device)).item()

    # return the loss
    return loss / len(data_loader)
    # return loss




# train model function
def train_model(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True, path_to_save_model=None):
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
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
            # fill in how to train your network using only the clean images

            # move to device
            x_clean = x_clean.to(device)
            label = label.to(device)

            # forward pass
            output = model(x_clean)
            loss = criterion(output, label)
        
            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()

        # calculate validation loss
        valid_loss = calculate_loss_ex4(model, valid_loader, criterion, device)

        # average loss for this epoch = train_loss / n_batches
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Average train loss for epoch {epoch} is {train_loss}, validation loss is {valid_loss}")

        # write the model parameters to a file every 5 epochs
        if write_to_file and epoch % 5 == 0:
            torch.save(model.state_dict(), f"{path_to_save_model}_{epoch}_epochs.pth")

    if write_to_file:
        torch.save(model.state_dict(), f"{path_to_save_model}_{epoch}_epochs.pth")

    # return the trained model
    return model, train_losses, valid_losses
