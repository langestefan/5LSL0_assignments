from re import X
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

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



def softthreshold(x, shrinkage):
    H, W = x.shape
    
    # compare each pixels in x with shrinkage value
    for i in range(H):
        for j in range(W):
            if np.abs(x[i,j]) > shrinkage:
                x[i,j] = ((np.abs(x[i,j]) - shrinkage)/np.abs(x[i,j]))*x[i,j]
            else:
                x[i,j] = 0

    return x

def ISTA(mu, shrinkage, K, y):
    H, W = y.shape[1:]
    A = np.identity(H)
    I = np.identity(H)

    # initialize 
    input_images = y + 1
    x_k = np.zeros((H, W))

    image_list = []

    for idx, y in tqdm(enumerate(input_images)):
        # print("shape of y: ", y.shape)

        for i in range(K):
            # gradient step, z = f1(x_k)
            z = np.dot((I - mu*np.dot(A.T, A)), x_k) + mu*np.dot(A, y)
        
            # soft thresholding
            x_k = softthreshold(z, shrinkage)

        # store the results
        image_list.append(x_k)
    
    # convert to tensor
    x_out = torch.from_numpy(np.array(image_list)).float()

    return x_out - 1


def main():
    # define parameters
    data_loc = 'data' #change the datalocation to something that works for you
    batch_size = 64

    # get dataloaders
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # samples of digits 0-10
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)

    x_clean_0_to_10 = x_clean_example[:10].squeeze()
    x_noisy_0_to_10 = x_noisy_example[:10].squeeze()

    print("input size: ", x_noisy_0_to_10.shape)

    # ISTA parameters working
    mu = 0.3
    shrinkage = 0.2
    K = 10

    # ISTA
    x_ista = ISTA(mu, shrinkage, K, x_noisy_0_to_10)
    print("ista output size: ", x_ista.shape)

    # compute MSE
    mse = torch.nn.MSELoss()
    mse_loss = mse(x_ista, x_clean_0_to_10)
    print("mse loss: ", mse_loss.item())

    # plot the results
    plot_examples(x_clean_0_to_10, x_noisy_0_to_10, x_ista)



if __name__ == "__main__":
    main()