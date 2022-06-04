from multiprocessing import reduction
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm as tqdm

# import torch mse_loss
from torch.nn.functional import mse_loss, binary_cross_entropy




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


def compute_final_loss(output_decoder, x_clean, x_mean, x_log_stddev, beta=1e-3):
    """
    Computes the final loss.
    -------
    output_decoder: torch.Tensor
        The output of the decoder
    x_clean: torch.Tensor of shape (batch_size, 1, 32, 32)
        The clean images
    x_mean: torch.Tensor
        The mean of the latent representation
    x_logvar: torch.Tensor
        The log variance of the latent representation
    beta: float
        The beta value
    """
    # MSE reconstruction loss
    reconstruction_loss = mse_loss(output_decoder, x_clean)

    # compute the KL loss
    x_var = torch.square(torch.exp(x_log_stddev))
    kl_loss = -0.5 * torch.sum(1 + 2*x_log_stddev - torch.square(x_mean) - x_var, dim=1)

    # return the average loss over all images in batch
    return torch.mean(reconstruction_loss + beta*kl_loss)

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
    
    # initialize the loss
    test_loss = 0
    test_kl_loss = 0


    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):

            image_batch = x_noisy if use_noisy_images else x_clean
            x_clean = x_clean.to(device)

            # network output
            output_decoder, x_sample, x_mean, x_log_stddev = model(x_clean)

            # compute the loss
            loss = compute_final_loss(output_decoder, x_clean, x_mean, x_log_stddev)
 
            # append latent representation and the output of minibatch to the list
            latent_list.append(x_sample.detach().cpu())
            output_list.append(output_decoder.detach().cpu())
            label_list.append(test_label.detach().cpu())

            # add loss to the total loss
            test_loss += loss.item()

        # print the average loss for this epoch
        test_losses = test_loss / len(test_loader)

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


def plot_loss(train_losses, validation_losses, save_path):
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
    ax.plot(validation_losses, label='Validation loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

def plot_kl_loss(kl_losses, save_path):
    """
    Plots the loss.
    -------
    train_losses: list
        The training loss
    valid_losses: list
        The validation loss
    """
    num_epochs = len(kl_losses)

    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(kl_losses, label='kl loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()




def train(model,optimizer,epochs,train_loader,test_loader,save_path=None):
    model.train()
    loss_train = 0.0
    loss_test = 0.0
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        # go over all minibatches
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
            # fill in how to train your network using only the clean images
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                model.to(device)

            # network output
            output_decoder, x_sample, x_mean, x_log_stddev = model(x_clean)

            # compute the loss
            loss = compute_final_loss(output_decoder, x_clean, x_mean, x_log_stddev)

            # print("test_kl_loss: {0} and reconst_loss: {1} ".format(train_kl_loss, reconst_loss))

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            # print('train_kl',model.encoder.kl)

        model.eval()

        # print mean and std of the latent space
        # print(torch.mean(x_mean), torch.mean(torch.exp(x_log_stddev)))

        with torch.no_grad():
            for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):

                # fill in how to train your network using only the clean images
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                    model.to(device)

                # network output
                output_decoder, x_sample, x_mean, x_log_stddev = model(x_clean)

                # compute the loss
                loss = compute_final_loss(output_decoder, x_clean, x_mean, x_log_stddev)

                loss_test += loss.item()
   
            print(f'train_loss = {loss_train/len(train_loader)}, test_loss = {loss_test/len(test_loader)}')

            # print("Mean, variance: ", x_mean, x_log_stddev)

        train_loss.append(loss_train/len(train_loader))
        test_loss.append(loss_test/len(test_loader))
        loss_train = 0.0
        loss_test = 0.0

    torch.save(model.state_dict(), f"{save_path}_{epoch+1}_best.pth")

    return model, x_sample, output_decoder, train_loss, test_loss