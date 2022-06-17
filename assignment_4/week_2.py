
import torch
from Fast_MRI_dataloader import create_dataloaders



if __name__ == "__main__":
# define parameters
    data_loc = 'assignment_4/Fast_MRI_Knee/' #change the datalocation to something that works for you
    batch_size = 2
    
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)