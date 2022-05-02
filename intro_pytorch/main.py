import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MNIST_dataloader import create_dataloaders
from model import build_model

def main():
    # define parameters
    data_loc = 'intro_pytorch/data' # change the datalocation to something that works for you
    batch_size = 64
    learning_rate = 0.01
    
    # get dataloader
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

    # define the model
    model = build_model()




# if the file is run as a script, run the main function
if __name__ == '__main__':
    main()
