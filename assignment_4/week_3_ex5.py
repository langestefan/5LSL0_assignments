import torch
import torch.nn as nn
import torch.optim as optim
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.fft import ifft2
import numpy as np
from train import load_model


class CNN_ex5(nn.Module):
    def __init__(self):
        super(CNN_ex5,self).__init__()
        # create layers here
        self.conv = nn.Sequential(
            # input is N, 1, 320, 320
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=1, padding=2), # N, 64, 320, 320
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2), # N, 64, 320, 320
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2), # N, 64, 320, 320
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(5, 5), stride=1, padding=2), # N, 1, 320, 320
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
       

    def forward(self, x):
       
        return self.conv(x)

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
    # go over all minibatches
    for i,(kspace, M, gt) in enumerate(tqdm(data_loader)):
        
        # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
        gt_unsqueeze = torch.unsqueeze(gt,dim =1)
        kspace_unsqueeze = torch.unsqueeze(kspace,dim =1)

        # get accelerated MRI image from partial k-space
        acc_mri = ifft2(kspace_unsqueeze)
        acc_mri = torch.log(torch.abs(acc_mri)+1e-20)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gt_unsqueeze, acc_mri = [x.cuda() for x in [gt_unsqueeze, acc_mri]]
            model.to(device)

        # forward pass
        x_out = model(acc_mri)
        
        # calculate loss
        loss += criterion(x_out, gt_unsqueeze).item()

    # return the loss
    return loss / len(data_loader)

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
        for i,(kspace, M, gt) in enumerate(tqdm(train_loader)):
            
            # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
            gt_unsqueeze = torch.unsqueeze(gt,dim =1)
            kspace_unsqueeze = torch.unsqueeze(kspace,dim =1)

            # get accelerated MRI image from partial k-space
            acc_mri = ifft2(kspace_unsqueeze)
            acc_mri = torch.log(torch.abs(acc_mri)+1e-20)

            # move to device
            gt_unsqueeze = gt_unsqueeze.to(device)
            acc_mri = acc_mri.to(device)
            model = model.to(device)

            # forward pass
            x_out = model(acc_mri)
            loss = criterion(x_out, gt_unsqueeze) 
        
            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()

        # calculate validation loss
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

def plot_ex5c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (10,10))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,0,:,:],vmin=-1.4,interpolation='nearest',cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Accelerated MRI')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,0,:,:],vmax=1.4,interpolation='nearest',cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Reconstruction from CNN')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Ground truth')

    #plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()
    


def plot_loss(train_losses, test_losses, save_path):
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
    ax.plot(test_losses, label='Testing loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 2))
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    data_loc = 'assignment_4/Fast_MRI_Knee/' #change the datalocation to something that works for you
    batch_size = 6

    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

    model = CNN_ex5()

    # train the model
    device = torch.device('cuda:0')
    # n_epochs = 10
    # learning_rate = 1e-4
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model, train_losses, test_losses, test_acc_mri, test_x_out, test_gt = train_model(model, train_loader, test_loader, optimizer, criterion, 
    #                                                 n_epochs, device, write_to_file=True, save_path='assignment_4/models/')
    
    # # # plot the loss for exercise 5b
    # plot_loss(train_losses, test_losses, 'assignment_4/figures/ex5b_loss.png')

    # # exercise 5c

    # load the trained model
    model = load_model(model, "assignment_4/models/_9_epochs.pth")
    for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
        if i == 1:
            break
    # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
    test_gt_unsqueeze = torch.unsqueeze(gt,dim =1)
    kspace_unsqueeze = torch.unsqueeze(kspace,dim =1)

    # get accelerated MRI image from partial k-space
    test_acc_mri = ifft2(kspace_unsqueeze)
    test_acc_mri = torch.log(torch.abs(test_acc_mri)+1e-20)

    # get reconstructed image from CNN
    test_x_out = model(test_acc_mri)
    # detach x_out from GPU
    test_x_out = test_x_out.detach().cpu().numpy()

    plot_ex5c(test_acc_mri, test_x_out, test_gt_unsqueeze, 'assignment_4/figures/ex5c.png')



    # exercise 5d 
    # The mean squared error between ground truth and CNN output is 0.0187.

