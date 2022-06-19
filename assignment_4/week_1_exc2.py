# %% imports
# libraries
from re import X
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
import MNIST_dataloader

# set torches random seed
torch.random.manual_seed(0)



def plot_examples(clean_images, noisy_images, ista_output, num_examples=10):
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

    print("input size: ", noisy_images.shape)
    print("output size: ", ista_output.shape)
    print("clean_images size: ", clean_images.shape)


    # show the examples in a plot
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.imshow(noisy_images[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(ista_output[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(clean_images[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    #plt.savefig("assignment_4/figures/exercise_1_b.png", dpi=300, bbox_inches='tight')
    plt.show()

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

def train_model(model, train_loader, n_epochs, optimizer, criterion):
    """ Train the model.
    Args:
        model (Model class): Untrained model to train.
        train_loader (DataLoader): DataLoader for training data.
        n_epochs (int): Number of epochs to train for.
    Returns:
        Model: Trained model.
    """
    model.train()
    train_losses = []

    for epoch in range(n_epochs):
        # go over all minibatches
        loss_train = 0.0
        loss = 0.0
        for batch_idx, (x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
           
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                model.to(device)

            x_out = model(x_noisy)
            loss = criterion(x_out, x_clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        train_losses.append(loss_train/len(train_loader))
        print(f'Epoch {epoch+1}/{n_epochs} Loss: {loss_train/len(train_loader)}')
 
    # save the trained model
    torch.save(model.state_dict(), f"assignment_4/models/{epoch+1}.pth")

    return model, train_losses


class W_2K_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1)    
        )

    def forward(self, x):
        return self.conv(x)

class W_2K(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1)    
        )

    def forward(self, x):
        return self.conv(x)


class LISTA(nn.Module):
    def __init__(self, n_unfolded_iter, lambda_init):
        super(LISTA, self).__init__()

        self.n_unfolded_iter = n_unfolded_iter
        self.shrinkage = nn.Parameter(torch.full((n_unfolded_iter,), lambda_init))

        # module lists for the unfolded iterations
        self.weight_2k_1 = nn.ModuleList([W_2K_1() for _ in range(self.n_unfolded_iter)])
        self.weight_2k   = nn.ModuleList([W_2K() for _ in range(self.n_unfolded_iter)])

    def _shrinkage(self, z, lambd):
        # smooth version of shrinkage
        x_smooth =  z + 0.5 * (torch.sqrt((z - lambd)**2 + 1) - torch.sqrt((z + lambd)**2 + 1))
    
        return x_smooth

    def forward(self, x):
        # initialize 
        y = x
        x_k = 0

        # iterate over the unfolded iterations
        for i in range(self.n_unfolded_iter):
            # output of the 2k-1 weight matrix
            y_2k_1 = self.weight_2k_1[i](y)

            # output of the 2k weight matrix
            z = self._shrinkage(y_2k_1 + x_k, lambd=self.shrinkage[i])
            x_k = self.weight_2k[i](z)       
  
        return x_k

def test_ex2b(model, x_noisy_test, x_clean_test):

    model.eval()
    x_out = model(x_noisy_test)
    x_model_out = x_out.detach().numpy()

    # take only the first 10 images and squeeze
    x_model_out = x_model_out[:10, :, :].squeeze()
    x_clean_test = x_clean_test[:10, :, :].squeeze()
    x_noisy_test = x_noisy_test[:10, :, :].squeeze()

    plot_examples(x_clean_test, x_noisy_test, x_model_out)


def main():
    # define parameters
    data_loc = 'data' #change the datalocation to something that works for you

    # get dataloaders
    batch_size = 64
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # samples of digits 0-10
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)

    x_clean_0_to_10 = x_clean_example[:10].squeeze()
    x_noisy_0_to_10 = x_noisy_example[:10].squeeze()
    labels_0_to_10 = labels_example[:10].squeeze()

    print("input size: ", x_noisy_0_to_10.shape)

    # generate LISTA model
    model = LISTA(n_unfolded_iter=3, lambda_init=0.5)
    print(model)

    # train the model
    n_epochs = 15
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model, train_losses = train_model(model, train_loader, n_epochs, optimizer, criterion)

    # load the trained model
    model = load_model(model, "assignment_4/models/LISTA_epoch15.pth")

    # exercise 2b
    test_ex2b(model, x_noisy_example, x_clean_example)




if __name__ == "__main__":
    main()