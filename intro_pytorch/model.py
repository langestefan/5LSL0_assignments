import torch
import torch.nn as nn


# Select CPU or GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:                         
    device = torch.device('cpu')

print("Using device:", device)

# AutoEncoder model class
class AutoEncoder(nn.Module): # CNN for image classification
    def __init__(self):
        super().__init__()
        
        # input is 1x28x28
        self.fc1 = nn.Sequential(    
            nn.Linear(28*28, 14*14)
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
            nn.Linear(14*14, 28*28)
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
    model.to(device)
    return model