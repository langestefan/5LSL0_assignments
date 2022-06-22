# %% imports
# libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2, ifftshift

import os

from zmq import device

from assignment_4.week_2_exc4 import get_accelerate_MRI
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
from train import train_model, plot_loss, plot_examples
from week_3_ex5 import CNN_ex5



# set torches random seed
torch.random.manual_seed(0)

class ProxNet(nn.Module):
    def __init__(self, n_unfolded_iter, mu_init):
        super(ProxNet, self).__init__()

        self.n_unfolded_iter = n_unfolded_iter
        self.mu = nn.Parameter(torch.full((n_unfolded_iter,), mu_init))

        # module lists for the unfolded iterations
        self.proximal_operator = nn.ModuleList([CNN_ex5() for _ in range(self.n_unfolded_iter)])
    
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

    def forward(self, k_space, M):
        
        # get accelerated MRI image from partial k-space
        y = get_accelerate_MRI(k_space)
        #y = torch.log(torch.abs(y)+1e-20)
    
        image_list = []
        for idx, (y, M) in enumerate(zip (y,M)):
            # iterate over the unfolded iterations
            for i in range(self.n_unfolded_iter):
                # output of the 2k-1 weight matrix
                y_2k_1 = self.weight_2k_1[i](y)

                # output of the 2k weight matrix
                z = self._shrinkage(y_2k_1 + x_k, lambd=self.shrinkage[i])
                x_k = self.weight_2k[i](z)       

           # print("shrinkage: ", self.shrinkage.data)
           
        return x_k


def main():
    # define parameters
    data_loc = 'assignment_4/Fast_MRI_Knee/' #change the datalocation to something that works for you
    batch_size = 6

    # get the dataloaders
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

 

    # generate LISTA model
    model = ProxNet(n_unfolded_iter=5, lambda_init=2.0)
    print(model)

    # train the model
    device = torch.device('cuda:0')
    n_epochs = 1
    learning_rate = 1e-4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, train_losses, valid_losses = train_model(model, train_loader, test_loader, optimizer, criterion, 
                                                    n_epochs, device, write_to_file=True, save_path='assignment_4/models/')


    # plot the losses for the training and validation
    plot_loss(train_losses, valid_losses, save_path='assignment_4/figures/' + 'losses_exc2a.png')

    # load the trained model
    # model = load_model(model, "assignment_4/models/LISTA_epoch15_v2.pth")

    # print shrinkage parameters
    print(model.mu.data)

    # exercise 2b
    # move the model to the cpu
    # model.cpu()
    # test_ex2b(model, x_noisy_example, x_clean_example)




if __name__ == "__main__":
    main()