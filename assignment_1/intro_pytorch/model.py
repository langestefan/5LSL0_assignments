import torch.nn as nn
import torch
import numpy as np

use_activation = True
latent_size = 8*8


# ReLU class wrapper from PyTorch nn.Module so we can easily use it in the model
# see: https://www.kaggle.com/code/aleksandradeis/extending-pytorch-with-custom-activation-functions/notebook
def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return torch.max(x, torch.zeros_like(x))

class customReLU(nn.Module):

    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return relu(input) # apply custom ReLU


# AutoEncoder model class
class AutoEncoder(nn.Module): 
    def __init__(self):
        super().__init__()
        
        # input is 1x32x32
        self.fc1 = nn.Sequential(    
            nn.Flatten(),
            nn.Linear(32*32, 14*14)
        )
        # dimensionality reduction 14x14 -> 8x8
        self.fc2 = nn.Sequential(  
            nn.Linear(14*14, 8*8)
        )
        # dimensionality increase 8x8 -> 14x14
        self.fc3 = nn.Sequential(    
            nn.Linear(8*8, 14*14)
        )        
        # output is 1x32x32
        self.fc4 = nn.Sequential(    
            nn.Linear(14*14, 32*32), 
        )    

        # # activation function
        # if use_activation:  
        #     self.activation = customReLU()
        # else:
        #     self.activation = nn.Identity()

        # self.fc1.append(self.activation)
        # self.fc2.append(self.activation)
        # self.fc3.append(self.activation)
        # # self.fc4.append(self.activation) # don't put ReLU on the output layer!!!
        # self.fc4.append(nn.Flatten())           
            
    def forward(self, x): 
        x = self.fc1(x) 
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        x = relu(x)
        x = self.fc4(x)   
        return x

def build_model():
    """
    Returns the autencoder model instance.
    -------
    model : model class
        Model structure to fit, as defined by build_model().
    """
    model = AutoEncoder()
    return model

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