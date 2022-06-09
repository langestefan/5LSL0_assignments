# %% imports
# libraries
from re import X
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch.optim as optim
# local imports
import MNIST_dataloader


# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
# x_clean_train = train_loader.dataset.Clean_Images
# x_noisy_train = train_loader.dataset.Noisy_Images
# labels_train  = train_loader.dataset.Labels

# x_clean_test  = test_loader.dataset.Clean_Images
# x_noisy_test  = test_loader.dataset.Noisy_Images
# labels_test   = test_loader.dataset.Labels

# # use these 10 examples as representations for all digits
# x_clean_example = x_clean_test[0:10,:,:,:]
# x_noisy_example = x_noisy_test[0:10,:,:,:]
# labels_example = labels_test[0:10]

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
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(ista_output[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    #plt.savefig("assignment_4/figures/exercise_1_b.png", dpi=300, bbox_inches='tight')
    plt.show()

# %% ISTA
def softthreshold(x,shrinkage):
    # x: torch.Size([32, 32])
    x_thx = x
    
    # compare each pixels in x with shrinkage value
    for i in range(32):
        for j in range(32):
            if x[i,j] > shrinkage:
                x_thx[i,j] = ((np.abs(x[i,j]) - shrinkage)/np.abs(x[i,j]))*x[i,j]
            else:
                x_thx[i,j] = 0

    return x_thx

def ISTA(mu,shrinkage,K,y):

    A = np.identity(32)
    I = np.identity(32)

    for i in tqdm(range(K)):

        if i == 0:
            input = y
        else:
            input = x_out
        image_list = []
        for z in range(len(input)):            
            x_old = input[z,0,:,:]
            x_ista = mu*np.dot(A,y[z,0,:,:]) + np.dot((I-mu*A*A.T),x_old)
            x_new = softthreshold(x_ista,shrinkage) 
            image_list.append(x_new)
           
        # convert to array
        x_out = np.array(image_list)
        
        # convert to tensor
        x_out = torch.from_numpy(x_out).float()
        x_out = x_out.unsqueeze(1)

    return x_out

class CNN_ex2(nn.Module):
    def __init__(self):
        super(CNN_ex2,self).__init__()
        self.conv = nn.Sequential(
        # input is N, 1, 32, 32
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1), # N, 1, 32, 32
        nn.BatchNorm2d(1)       
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x

def smoother_counter(x,shrinkage_ex2):
    # x: torch.Size([64, 1, 32, 32])
    x_smooth = x
    
    # compare each pixels in x with shrinkage value
    for z in range(len(x)):
        for i in range(32):
            for j in range(32):
                x_smooth[z,0,i,j] = x[z,0,i,j] + 0.5 * ((torch.sqrt(((x[z,0,i,j] - shrinkage_ex2)**2) + 1))-(torch.sqrt(((x[z,0,i,j] + shrinkage_ex2)**2) + 1)))

    return x_smooth

def LISTA(model,unfolded_iterations,shrinkage_ex2,x_noisy):
    for i in range(unfolded_iterations):
        if i == 0:
            iter_out = 0
        else:
            iter_out = x_out3

        x_out1 = model(x_noisy) + iter_out      
        x_out2 = smoother_counter(x_out1,shrinkage_ex2[i])
        x_out3 = model(x_out2)

    return x_out3  

if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    
    # define parameters
    data_loc = 'assignment_4/data' #change the data location to something that works for you
    batch_size = 64
    mu = 0.9
    shrinkage = 0.1
    K = 2
    # parameters for ex2
    shrinkage_ex2 = [0.1,0.2,0.3]
    n_epochs = 1
    learning_rate = 0.1
    unfolded_iterations = 3
    # create the optimizer
    model = CNN_ex2()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

    # # define the device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
    # print("Using device:", device)
   
    # get dataloader
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
    # take first 10 images from clean test data set: torch.Size([10, 1, 32, 32])
    x_clean_test  = test_loader.dataset.Clean_Images[0:10]
    # take first 10 images from noisy test data set: torch.Size([10, 1, 32, 32])
    x_noisy_test  = test_loader.dataset.Noisy_Images[0:10]
 
    

  
    # #start timer
    # start_time = time.time()
    # #exercise 1 a
    # x_ista = ISTA(mu,shrinkage,K,x_noisy_test)

    # #exercise 1 c
    # ISTA_mse_losses = 0
    # loss = 0
    # for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
        
    #     x_ista = ISTA(mu,shrinkage,K,x_noisy)
    #     loss = torch.nn.functional.mse_loss(x_ista,x_noisy)
    #     loss = np.array(loss)
    #     ISTA_mse_losses += loss
    #     print ("loss:",loss)
    #     print ("mse_loss:",ISTA_mse_losses)
   
    # print(f'test_loss = {ISTA_mse_losses/len(test_loader)}') # ISTA_mse_loss = 1.0228046873572525

    # print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # #exercise 1 b
    # plot_examples(x_clean_test, x_noisy_test, x_ista)

    # #exercise 2a

    # #start timer
    start_time = time.time()

    model.train()
    loss_train = 0.0
    train_loss = []
    for epoch in range(n_epochs):
        # go over all minibatches
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
            # # fill in how to train your network using only the clean images
            # if torch.cuda.is_available():
            #     device = torch.device('cuda:0')
            #     x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
            #     model.to(device)

            x_out = LISTA(model,unfolded_iterations,shrinkage_ex2,x_noisy)
            loss = criterion(x_clean, x_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        train_loss.append(loss_train/len(train_loader))
        print(f'Epoch {epoch+1}/{n_epochs} Loss: {loss_train/len(train_loader)}')

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

 

   
   