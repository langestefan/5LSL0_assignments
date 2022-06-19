
import torch
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2

def get_k_space(input) :
    # get the k-space
    k_space = fftshift(fft2(input))
    return k_space


if __name__ == "__main__":
# define parameters
    data_loc = 'assignment_4/Fast_MRI_Knee/' #change the datalocation to something that works for you
    batch_size = 2
    
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)

    # go over the dataset
    for i,(kspace, M, gt) in enumerate(tqdm(test_loader)):
       
        k_space = get_k_space(gt)
        

    kspace_plot_friendly = torch.log(torch.abs(k_space[0,:,:])+1e-20)

    plt.figure(figsize = (10,10))

    plt.subplot(1,2,1)
    plt.imshow(kspace_plot_friendly,vmin=-2,interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('k_space')

    plt.subplot(1,2,2)
    plt.imshow(gt[0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('ground truth')

    plt.savefig("assignment_4/figures/3a.png",dpi=300,bbox_inches='tight')
    plt.show()

