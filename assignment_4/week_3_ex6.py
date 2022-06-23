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
from week_3_ex5 import CNN_ex5,plot_loss
from train import load_model

# set torches random seed
torch.random.manual_seed(0)

# element wise multiplication of k-space and M
def apply_mask_k_space(full_k_space, M):
    """
    Element wise multiplication of k-space and M
    -------
    full_k_space: torch.Tensor
        full k-space of MRI image
    M: torch.Tensor
        mask
    """
    assert full_k_space.dim() == 3, "full_k_space must be a 3D tensor"

    partial_k_space = torch.mul(full_k_space, M)
    return  partial_k_space

# convert k-space to MRI image
def kspace_to_mri(k_space, reverse_shift):
    """
    Convert k-space to MRI image
    -------
    k_space: torch.Tensor
        k-space of MRI image
    """
    assert k_space.dim() == 3, "k_space must be a 3D tensor"

    if reverse_shift:
        k_space = ifftshift(k_space, dim=(1, 2))

    MRI_image = ifft2(k_space, dim=(1, 2))
    MRI_image = torch.abs(MRI_image)

    return MRI_image

# convert MRI image into k-space
def mri_to_kspace(mri_image, apply_shift=True):
    """
    Convert MRI image into k-space.
    -------
    MRI_image: torch.Tensor (N, H, W) 
        Batch of MRI images. N is the batch size, (H, W) is the image size.
    """
    assert mri_image.dim() == 3, "mri_image must be a 3D tensor"
    k_space = fft2(mri_image, dim=(1, 2))

    if apply_shift:
        k_space = fftshift(k_space, dim=(1, 2))

    return k_space

class ProxNet(nn.Module):
    def __init__(self, n_unfolded_iter, mu_init):
        super(ProxNet, self).__init__()

        self.n_unfolded_iter = n_unfolded_iter
        self.mu = nn.Parameter(torch.full((n_unfolded_iter,), mu_init))

        # module lists for the unfolded iterations
        self.proximal_operator = nn.ModuleList([CNN_ex5() for _ in range(self.n_unfolded_iter)])
    
    def forward(self, partial_k_space, M):
        '''
        input partial_k_space, M is mask
        '''

        # convert partial_k_space to MRI image
        y = kspace_to_mri(partial_k_space, reverse_shift=False)
        

        # initialize ,
        x_t = y
        FY = partial_k_space
        # iterate over the unfolded iterations
        for i in range(self.n_unfolded_iter):
            
            # convert MRI image into k-space
            F_x = mri_to_kspace(x_t, apply_shift=True)
            

            # data consistency term abs(Finv(FX - mu*M*FX + mu*FY))
            z = F_x - self.mu[i] * apply_mask_k_space(F_x, M) + self.mu[i] * FY
            
            # convert k-space to MRI image
            x_t = kspace_to_mri(z, reverse_shift=False)

            # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
            x_t = torch.unsqueeze(x_t,dim =1)

            # apply the proximal operator
            x_t = self.proximal_operator[i](x_t)

            # squeeze to reduce channel dimension, (N, 1, 320, 320) -> (N, 320, 320)
            x_t = torch.squeeze(x_t,dim =1)

        #print("mu: ", self.mu.data)
       
        return x_t

def plot_ex6c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (12,12))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Accelerated MRI')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,:,:],vmax=2,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Reconstruction from ProxNet')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
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
    for i,(partial_kspace, M, gt) in enumerate(tqdm(data_loader,position=0, leave=False, ascii=False)):
  
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gt, partial_kspace, M = [x.cuda() for x in [gt, partial_kspace, M]]
            model.to(device)

        # forward pass
        ProxNet_out = model(partial_kspace,M)
        
        # calculate loss
        loss += criterion(ProxNet_out, gt).item()
  
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

        for idx,(partial_kspace, M, gt) in enumerate(tqdm(train_loader, position=0, leave=False, ascii=False)):
            
            # move to device
            gt = gt.to(device)
            M = M.to(device)
            partial_kspace = partial_kspace.to(device)
            model = model.to(device)

            # forward pass
            ProxNet_out = model(partial_kspace,M)
            loss = criterion(ProxNet_out, gt) 
        
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
            torch.save(model.state_dict(), f"{save_path}ProxNet_{epoch}_epochs.pth")

    if write_to_file:
        torch.save(model.state_dict(), f"{save_path}ProxNet_{epoch}_epochs.pth")

    # return the trained model
    return model, train_losses, valid_losses

def main():
    # define parameters
    data_loc = 'assignment_4/Fast_MRI_Knee/' # change the datalocation to something that works for you
    batch_size = 6
    # get the dataloaders
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

    model = ProxNet(n_unfolded_iter=5, mu_init=0.5)
    #print(model)

    # train the model
    device = torch.device('cuda:0')
    # n_epochs = 20
    # learning_rate = 1e-4
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    # # train the model
    # model, train_losses, test_losses = train_ex6c(model, train_loader, test_loader, optimizer, criterion, 
    #                                                 n_epochs, device, write_to_file=True, save_path='assignment_4/models/')
    # # # plot the loss for exercise 6b
    # plot_loss(train_losses, test_losses, 'assignment_4/figures/ex6b_loss.png')


    # load the trained model
    model = load_model(model, "assignment_4/models/ProxNet_19_epochs.pth")
    for i,(partial_kspace, M, gt) in enumerate(tqdm(test_loader)):
        if i == 1:
            break
 
    # get reconstructed image from ProxNet
    test_x_out = model(partial_kspace,M)
    # detach x_out from GPU
    test_x_out = test_x_out.detach().cpu().numpy()

    # get input accelerated MRI iamge
    accelerated_MRI = kspace_to_mri(partial_kspace,reverse_shift=True)

    plot_ex6c(accelerated_MRI, test_x_out, gt, 'assignment_4/figures/ex6c.png')



    # exercise 6d 
    # The mean squared error between ground truth and ProxNet output for entrie test dataset is 0.01634212054039647.
    


if __name__ == "__main__":
    main()
    