import torch.nn as nn
import torch

# AutoEncoder model class
class AutoEncoder(nn.Module): 
    def __init__(self):
        super().__init__()
        
        # input is 1x32x32
        self.fc1 = nn.Sequential(    
            nn.Flatten(),
            nn.Linear(32*32, 14*14)
        )
        # dimensionality reduction 14x14 -> 7x7
        self.fc2 = nn.Sequential(  
            nn.Linear(14*14, 7*7)
        )
        # dimensionality increase 7x7 -> 14x14
        self.fc3 = nn.Sequential(    
            nn.Linear(7*7, 14*14)
        )        
        # output is 1x28x28
        self.fc4 = nn.Sequential(    
            nn.Linear(14*14, 32*32),               
            nn.Flatten()
        )      
    
    def forward(self, x): 
        x = self.fc1(x) 
        x = self.fc2(x)
        x = self.fc3(x)
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