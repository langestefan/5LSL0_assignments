import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm as tqdm
import time



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
    reconstruction_term_weight = 1
    test_loss = 0

    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):

            image_batch = x_noisy if use_noisy_images else x_clean
            x_clean = x_clean.to(device)

            # forward pass
            output_decoder, x_sample, x_mean, x_log_var = model(x_clean)
            # total loss = reconstruction loss + KL divergence
            # kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            train_kl_loss = -0.5 * torch.sum(1 + x_log_var 
                                      - x_mean**2 
                                      - torch.exp(x_log_var), 
                                      axis=1) # sum over latent dimension

            train_kl_loss = train_kl_loss.mean() # average over batch dimension
    
            test_loss = criterion(output_decoder, x_clean)
            test_loss = test_loss.sum() # sum over pixels
            test_loss = test_loss.mean() # average over batch dimension
            
            test_batch_loss = reconstruction_term_weight*test_loss + train_kl_loss
            # append latent representation and the output of minibatch to the list
            latent_list.append(x_sample.detach().cpu())
            output_list.append(output_decoder.detach().cpu())
            label_list.append(test_label.detach().cpu())

            # add loss to the total loss
            test_batch_loss += test_batch_loss.item()

        # print the average loss for this epoch
        test_losses = test_batch_loss / len(test_loader)

        print(f"Test loss is {test_losses}.")
        
        
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


def plot_loss(train_kl_losses, batch_losses, save_path):
    """
    Plots the loss.
    -------
    train_losses: list
        The training loss
    valid_losses: list
        The validation loss
    """
    num_epochs = len(train_kl_losses)

    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_kl_losses, label='Training kl loss')
    ax.plot(batch_losses, label='Train batch loss')
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
    # set model to training mode
    model.train()
    # to keep track of loss
    train_kl_losses = []
    train_reconstruction_losses = []
    train_epoch_losses = []


    start_time = time.time()

    # go over all epochs
    for epoch in range(n_epochs):
        print(f"\nTraining Epoch {epoch}:")
        train_kl_loss = 0
        train_reconstruction_loss = 0
        train_batch_loss = 0
        # go over all minibatches
        for batch_idx,(x_clean, __, label) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
            # fill in how to train your network using only the clean images

            # move to device
            x_clean = x_clean.to(device)
            # label = label.to(device)

            # forward pass
            output_decoder, x_sample, x_mean, x_log_var = model(x_clean)
            
            # calculate loss
            #train_reconstruction_loss = criterion(output_decoder, x_clean).sum()
            train_reconstruction_loss = ((output_decoder - x_clean)**2).sum()
            train_kl_loss = 0.5 * (x_mean**2 + x_log_var**2 - torch.log(x_log_var)- 1).sum()
            # print("x_mean :",x_mean)
            # print("x_log_var :",x_log_var)
            # print("kl loss :",train_kl_loss)


            train_batch_loss = train_reconstruction_loss + train_kl_loss
        
            # backward pass, update weights
            optimizer.zero_grad()
            train_batch_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # add loss to the total loss
            train_kl_loss += train_kl_loss.item()
            train_reconstruction_loss += train_reconstruction_loss.item()
            train_batch_loss += train_batch_loss.item()
            

        # average loss for this epoch = train_loss / n_batches
        train_kl_loss = train_kl_loss / len(train_loader)
        train_reconstruction_loss = train_reconstruction_loss / len(train_loader)
        train_batch_loss = train_batch_loss / len(train_loader)

        # append to list of losses
        train_kl_losses.append(train_kl_loss)
        train_reconstruction_losses.append(train_reconstruction_loss)
        train_epoch_losses.append(train_batch_loss)

        print(f"Average batch train loss for epoch {epoch+1} is {train_batch_loss}.kl loss is {train_kl_loss} and reconstruction loss is {train_reconstruction_loss}")

        # write the model parameters to a file every 5 epochs
        if write_to_file and epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_{epoch+1}_epochs.pth")

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    if write_to_file:
        torch.save(model.state_dict(), f"{save_path}_{epoch+1}_epochs.pth")

    # return the trained model
    return model, train_kl_losses, train_reconstruction_losses, train_epoch_losses
