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
    # y: torch.Size([64, 1, 32, 32])

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

class LISTA(nn.Module):
    def __init__(self):
        super(LISTA,self).__init__()
        self.conv1 = nn.Sequential(
        # input is N, 1, 32, 32
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1), # N, 1, 32, 32
        nn.BatchNorm2d(1)       
        )
        self.conv2 = nn.Sequential(
        # input is N, 1, 32, 32
        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1), # N, 1, 32, 32
        #nn.BatchNorm2d(1)       
        )
    
    def smoother_counter(self, x_out1, shrinkage_ex2):
        # x_out1: torch.Size([64, 1, 32, 32])
        x_smooth =  x_out1 + 0.5 * ((torch.sqrt(((x_out1 - shrinkage_ex2)**2) + 1))-(torch.sqrt(((x_out1 + shrinkage_ex2)**2) + 1)))
    
        return x_smooth
        
    def forward(self,x):
        unfolded_iterations = 3
        shrinkage_ex2 = [1,2,3]
        for i in range(unfolded_iterations):
            if i == 0:
                iter_out = 0
            else:
                iter_out = x_out3

            x_out1 = self.conv1(x) + iter_out      
            x_out2 = self.smoother_counter(x_out1,shrinkage_ex2[i])
            x_out3 = self.conv2(x_out2)
      
        return x_out2


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
    loss_train = 0.0
    train_loss = []
    for epoch in range(n_epochs):
        # go over all minibatches
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
           
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                model.to(device)

            x_out = model(x_noisy)
            loss = criterion(x_clean, x_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        train_loss.append(loss_train/len(train_loader))
        print(f'Epoch {epoch+1}/{n_epochs} Loss: {loss_train/len(train_loader)}')
    # save the trained model
    torch.save(model.state_dict(), f"assignment_4/models/{epoch+1}.pth")

    return model, train_loss

def test_model(model, x_noisy_test):

    model.eval()
    x_out = model(x_noisy_test)
    x_out = x_out.detach().numpy()
    plot_examples(x_clean_test, x_noisy_test, x_out)


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)
    
    # define parameters
    data_loc = 'assignment_4/data' #change the data location to something that works for you
    batch_size = 64
    mu = 0.9
    shrinkage = 0.1
    K = 1

    # parameters for ex2
    n_epochs = 2
    learning_rate = 0.1

    model = LISTA()
 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5) 

    # get dataloader
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
    # take first 10 images from clean test data set: torch.Size([10, 1, 32, 32])
    x_clean_test  = test_loader.dataset.Clean_Images[0:10]
    # take first 10 images from noisy test data set: torch.Size([10, 1, 32, 32])
    x_noisy_test  = test_loader.dataset.Noisy_Images[0:10]
    




    # #start timer
    # start_time = time.time()
    # #exercise 1 a
    # #x_ista = ISTA(mu,shrinkage,K,x_noisy_test)

    # #exercise 1 c
    # ISTA_mse_losses = 0
    # loss = 0
    # for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
        
    #     x_ista = ISTA(mu,shrinkage,K,x_noisy)
    #     loss = torch.nn.functional.mse_loss(x_ista,x_clean)
    #     loss = np.array(loss)
    #     ISTA_mse_losses += loss
    #     print ("loss:",loss)
    #     print ("mse_loss:",ISTA_mse_losses)
   
    # print(f'test_loss = {ISTA_mse_losses/len(test_loader)}') 
    # # ISTA_mse_loss_nosiy = 1.0228
    # # ISTA_mse_loss_clean = 0.8412


    # print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # #exercise 1 b
    # plot_examples(x_clean_test, x_noisy_test, x_ista)

    # exercise 2a

    # train the model
    # start timer
    start_time = time.time()

    #model, train_loss = train_model(model, train_loader, n_epochs, optimizer, criterion)

    # load the trained model
    model = load_model(model, "assignment_4/models/10.pth")
    test_model(model, x_noisy_test)

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

 

   
   