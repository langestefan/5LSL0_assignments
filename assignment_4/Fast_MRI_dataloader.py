# %% imports
# libraries
import torch
from torch.utils.data import Dataset,DataLoader
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# %% Fast MRI dataset
class Fast_MRI(Dataset):
    # initialization of the dataset
    def __init__(self, split,data_loc):
        # save the input parameters
        self.split    = split 
        self.data_loc = data_loc
        
        # get all the files
        self.file_names = glob.glob(f"{data_loc}//{split}//*.npz")
    
    # return the number of examples in this dataset
    def __len__(self):
        return len(self.file_names)*5
    
    # create a a method that retrieves a single item form the dataset
    def __getitem__(self, idx):
        file_name = self.file_names[idx//5]
        data = np.load(file_name)
        
        kspace = data['kspace']
        M = data['M']
        gt = data['gt']
        
        # get one of 3 slices
        kspace = kspace[idx%5,:,:]
        gt = gt[idx%5,:,:]
        
        return kspace, M, gt

    
# %% dataloader for the Fast MRI dataset
def create_dataloaders(data_loc, batch_size):
    dataset_train = Fast_MRI("train", data_loc)
    dataset_test  = Fast_MRI("test" , data_loc)
    
    Fast_MRI_train_loader =  DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  drop_last=False)
    Fast_MRI_test_loader  =  DataLoader(dataset_test , batch_size=batch_size, shuffle=True, drop_last=False)
    
    return Fast_MRI_train_loader, Fast_MRI_test_loader

# %% test if the dataloaders work
if __name__ == "__main__":
    # define parameters
    data_loc = 'D://5LSL0-Datasets//Fast_MRI_Knee' #change the datalocation to something that works for you
    batch_size = 2
    
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    
    # go over the dataset
    for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
        continue
    
    # %% plot the last example
    kspace_plot_friendly = torch.log(torch.abs(kspace[0,:,:])+1e-20)
    
    plt.figure(figsize = (10,10))
    plt.subplot(1,3,1)
    plt.imshow(kspace_plot_friendly,vmin=-2.3,interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('partial k-space')
    
    plt.subplot(1,3,2)
    plt.imshow(M[0,:,:],interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('measurement mask')
    
    plt.subplot(1,3,3)
    plt.imshow(gt[0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('ground truth')
    
    plt.savefig("example.png",dpi=300,bbox_inches='tight')
    plt.close()