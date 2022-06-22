# %% imports
# libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2, ifftshift
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# local imports
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# local imports
from Fast_MRI_dataloader import create_dataloaders
from week_3_ex5 import CNN_ex5

# set torches random seed
torch.random.manual_seed(0)

def get_k_space(MRI_image) :
    # convert MRI image into k-space
    #k_space = fftshift(fft2(MRI_image))
    k_space = fft2(MRI_image)
    return k_space

def get_partial_k_space(k_space,M) :
    # element wise multiplication of k-space and M
    return  torch.mul(k_space, M)

def get_accelerate_MRI(k_space) :
    # convert k-space to MRI image
    return ifft2(k_space)

def get_accelerate_MRI_final(input) :
    # convert k-space to MRI image
    return ifft2(ifftshift(input))

class ProxNet(nn.Module):
    def __init__(self, n_unfolded_iter, mu_init):
        super(ProxNet, self).__init__()

        self.n_unfolded_iter = n_unfolded_iter
        self.mu = nn.Parameter(torch.full((n_unfolded_iter,), mu_init))

        # module lists for the unfolded iterations
        self.proximal_operator = nn.ModuleList([CNN_ex5() for _ in range(self.n_unfolded_iter)])
    
    def forward(self, y, M):

        # initialize 
        x_t = y
        # iterate over the unfolded iterations
        for i in range(self.n_unfolded_iter):

            F_x = get_k_space(x_t)
            k_space_y = get_k_space(y)

            z = F_x - self.mu[i] * get_partial_k_space(F_x, M) + self.mu[i] * k_space_y
            
            if i == self.n_unfolded_iter:
                x_t = get_accelerate_MRI_final(z)
            else:
                x_t = get_accelerate_MRI(z)

            x_t = torch.abs(x_t)
            x_t = self.proximal_operator[i](x_t)

        #print("mu: ", self.mu.data)
       
        return x_t

def plot_ex6c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (12,12))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,0,:,:],vmin=-1.5, vmax=2, interpolation='bilinear',cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Accelerated MRI')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,0,:,:],vmax=2.3, interpolation='nearest',cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Reconstruction from ProxNet')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Ground truth')

    #plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
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
    # go over all minibatches
    for i,(partial_kspace, M, gt) in enumerate(tqdm(data_loader)):
        
        # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
        gt_unsqueeze = torch.unsqueeze(gt,dim =1)
        par_kspace_unsqueeze = torch.unsqueeze(partial_kspace,dim =1)
        M_unsqueeze = torch.unsqueeze(M,dim =1)

        # get accelerated MRI image from partial k-space
        acc_mri = ifft2(par_kspace_unsqueeze)
        #acc_mri = torch.abs(acc_mri)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gt_unsqueeze, acc_mri, M_unsqueeze  = [x.cuda() for x in [gt_unsqueeze, acc_mri, M_unsqueeze]]
            model.to(device)

        # forward pass
        x_out = model(acc_mri,M_unsqueeze)
        
        # calculate loss
        loss += criterion(x_out, gt_unsqueeze).item()
        if i == 2:
            break
    # return the loss
    return loss / len(data_loader)

def train_ex6c(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True, save_path=None):
    # to keep track of loss
    train_losses = []
    valid_losses = []
    # go over all epochs
    for epoch in range(n_epochs):
        print(f"\nTraining Epoch {epoch}:")

        train_loss = 0
        valid_loss = 0

        for idx,(partial_kspace, M, gt) in enumerate(tqdm(train_loader)):
            
            # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
            gt_unsqueeze = torch.unsqueeze(gt,dim =1)
            partial_kspace_unsqueeze = torch.unsqueeze(partial_kspace,dim =1)
            M_unsqueeze = torch.unsqueeze(M,dim =1)

            # get accelerated MRI image from partial k-space
            acc_mri = ifft2(partial_kspace_unsqueeze)
            #acc_mri = torch.abs(acc_mri)

            # move to device
            gt_unsqueeze = gt_unsqueeze.to(device)
            acc_mri = acc_mri.to(device)
            model = model.to(device)
            M_unsqueeze = M_unsqueeze.to(device)

            # forward pass
            x_out = model(acc_mri,M_unsqueeze)
            loss = criterion(x_out, gt_unsqueeze) 
        
            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            train_loss += loss.item()
            if idx == 2:
                break
        # calculate validation loss
        valid_loss = calculate_loss(model, valid_loader, criterion, device) # autoencoder 
        # average loss for this epoch = train_loss / n_batches
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Average train loss for epoch {epoch} is {train_loss}, validation loss is {valid_loss}")

        # write the model parameters to a file every 5 epochs
        if write_to_file and epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}ProxNet_{epoch}_epochs.pth")

    if write_to_file:
        torch.save(model.state_dict(), f"{save_path}ProxNet_{epoch}_epochs.pth")

    # return the trained model
    return model, train_losses, valid_losses

def main():
    # define parameters
    data_loc = 'assignment_4/Fast_MRI_Knee/' # change the datalocation to something that works for you
    batch_size = 2

    # get the dataloaders
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

    model = ProxNet(n_unfolded_iter=5, mu_init=2.0)
    #print(model)

    # train the model
    device = torch.device('cuda:0')
    n_epochs = 1
    learning_rate = 1e-4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    # train the model
    model, train_losses, test_losses = train_ex6c(model, train_loader, test_loader, optimizer, criterion, 
                                                    n_epochs, device, write_to_file=True, save_path='assignment_4/models/')
  
    


if __name__ == "__main__":
    main()
    