from rich import reconfigure
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm as tqdm




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
    
    # initialize the loss
    test_loss = 0
    test_kl_loss = 0


    model.eval()

    # go over all minibatches
    with torch.no_grad():
        for batch_idx,(x_clean, x_noisy, test_label) in enumerate(tqdm(test_loader, position=0, leave=False, ascii=False)):

            image_batch = x_noisy if use_noisy_images else x_clean
            x_clean = x_clean.to(device)

            output_decoder, x_sample, x_mean, x_log_var = model(x_clean)
            test_kl_loss = 0.5 * (x_mean**2 + x_log_var - torch.log(x_log_var)- 1).sum()
            loss = ((output_decoder-x_clean)**2).sum() + test_kl_loss
            
 
            # append latent representation and the output of minibatch to the list
            latent_list.append(x_sample.detach().cpu())
            output_list.append(output_decoder.detach().cpu())
            label_list.append(test_label.detach().cpu())

            # add loss to the total loss
            test_loss += loss.item()

        # print the average loss for this epoch
        test_losses = test_loss / len(test_loader)/1e4

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

def train_ex7(model,optimizer,epochs,train_loader,test_loader,save_path=None):
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
            output_decoder, x_sample, x_mean, x_log_var = model(x_clean)
            train_kl_loss = 0.5 * (x_mean**2 + x_log_var - torch.log(x_log_var)- 1).sum()
            loss = ((output_decoder-x_clean)**2).sum() + train_kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            # print('train_kl',model.encoder.kl)

        model.eval()
        with torch.no_grad():
            for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
                # fill in how to train your network using only the clean images
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                    model.to(device)
                output_decoder, x_sample, x_mean, x_log_var = model(x_clean)
                test_kl_loss = 0.5 * (x_mean**2 + x_log_var - torch.log(x_log_var)- 1).sum()
                loss = ((output_decoder-x_clean)**2).sum() + test_kl_loss
                loss_test += loss.item()
   
            print(f'train_loss = {loss_train/len(train_loader)/1e4}, test_loss = {loss_test/len(test_loader)/1e4}')

        train_loss.append(loss_train/len(train_loader)/1e4)
        test_loss.append(loss_test/len(test_loader)/1e4)
        loss_train = 0.0
        loss_test = 0.0

    torch.save(model.state_dict(), f"{save_path}_{epoch+1}_best.pth")

    return model, x_sample, output_decoder, train_loss, test_loss

def train_ex8(model,optimizer,epochs,train_loader,test_loader,save_path=None):
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
            output_decoder, x_sample, x_mean, x_log_var = model(x_noisy)
            train_kl_loss = 0.5 * (x_mean**2 + x_log_var - torch.log(x_log_var)- 1).sum()
            loss = ((output_decoder-x_noisy)**2).sum() + train_kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            # print('train_kl',model.encoder.kl)

        model.eval()
        with torch.no_grad():
            for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
                # fill in how to train your network using only the clean images
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                    model.to(device)
                output_decoder, x_sample, x_mean, x_log_var = model(x_noisy)
                test_kl_loss = 0.5 * (x_mean**2 + x_log_var - torch.log(x_log_var)- 1).sum()
                loss = ((output_decoder-x_noisy)**2).sum() + test_kl_loss
                loss_test += loss.item()
   
            print(f'train_loss = {loss_train/len(train_loader)/1e4}, test_loss = {loss_test/len(test_loader)/1e4},epoch = {epoch}')

        train_loss.append(loss_train/len(train_loader)/1e4)
        test_loss.append(loss_test/len(test_loader)/1e4)
        loss_train = 0.0
        loss_test = 0.0

    torch.save(model.state_dict(), f"{save_path}_{epoch+1}_best.pth")

    return model, x_sample, output_decoder, train_loss, test_loss