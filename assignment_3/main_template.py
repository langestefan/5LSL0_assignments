# libraries
from turtle import shape
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from sklearn.decomposition import PCA

# from tqdm import tqdm
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import autoencoder_template
import train

def scatter_plot(model, x_clean_test, labels_train):



    # PCA  
    #vector_feature= [ fv.detach().numpy() for fv in vector_feature]                                                    
    pca = PCA(n_components=2)
    vector_feature = pca.fit_transform(latent)
     
    
    all_points = []
    for i in range(10):
        all_point = []
        for index,label in enumerate(labels_train):
            if label.item()==i and len(all_point)<20:
                all_point.append(vector_feature[index].tolist())
        all_points.append(all_point)   
    
    colors = plt.cm.Paired(np.linspace(0,1,len(all_points)))
    fig, ax = plt.subplots()
    for (points, color, digit) in zip (all_points, colors, range (10)):
        ax.scatter([item[0] for item in points],
                    [item[1] for item in points],
                    color = color, label = 'digit{}'.format(digit))
    ax.grid(True)
    plt.title('scatter plot') 
    ax.legend(loc='best')
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

def plot_images_exercise_1(x_data, recon):

    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_data[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2,10,i+11)
        plt.imshow(recon[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    #plt.savefig("exercise_1.png",dpi=300,bbox_inches='tight')
    plt.show() 

# set torches random seed
torch.random.manual_seed(0)

# define parameters
data_loc = 'data' #change the data location to something that works for you
batch_size = 64
n_epochs = 1
learning_rate = 3e-4

# get dataloader
train_loader,valid_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

x_clean_test  = test_loader.dataset.Clean_Images

labels_test   = test_loader.dataset.Labels

# create the autoencoder
AE = autoencoder_template.AE()

# load the trained model 
AE = load_model(AE, "AE_model_params.pth")

# create the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), learning_rate, weight_decay=1e-5)



# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
print("Using device:", device)

# move model to device
AE.to(device)

# %% training loop
AE, train_losses, valid_losses = train.train_model (AE, train_loader, valid_loader, optimizer, criterion, n_epochs, device, write_to_file=True)
#scatter_plot(AE, x_clean_test, labels_test)

""" # # move back to cpu    
recon = recon.detach().cpu()
latent = latent.detach().cpu()
x_clean = x_clean.detach().cpu()
x_noisy = x_noisy.detach().cpu()

batch_size_TEST = 1

# get X_clean_exsample
x_clean_test  = test_loader.dataset.Clean_Images[0:10]

recon_test, latent_test = AE(x_clean_test.to(device))
recon_test = recon_test.detach().cpu()

#plot_images_exercise_1(x_clean_test, recon_test)
#plot_images_exercise_1(x_clean, recon)


#train.plot_loss(train_losses=train_losses, valid_losses=valid_losses) """

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
""" 
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels """

# use these 10 examples as representations for all digits
""" 
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10] """