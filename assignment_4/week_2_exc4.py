from sqlalchemy import true
import torch
from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np

from torch.functional import F


# element wise multiplication of k-space and M
def apply_mask_k_space(full_k_space, M):
    """
    Element wise multiplication of k-space and M
    -------
    full_k_space: torch.Tensor
        full k-space of MRI image
    M: torch.Tensor
        mask

    """
    assert full_k_space.dim() == 3, "full_k_space must be a 3D tensor"

    partial_k_space = torch.mul(full_k_space, M)
    return  partial_k_space

# convert k-space to MRI image
def kspace_to_mri(k_space, reverse_shift):
    """
    Convert k-space to MRI image
    -------
    k_space: torch.Tensor
        k-space of MRI image
    """
    assert k_space.dim() == 3, "k_space must be a 3D tensor"

    if reverse_shift:
        k_space = ifftshift(k_space, dim=(1, 2))

    MRI_image = ifft2(k_space, dim=(1, 2))
    MRI_image = torch.abs(MRI_image)

    return MRI_image

# convert MRI image into k-space
def mri_to_kspace(mri_image, apply_shift=True):
    """
    Convert MRI image into k-space.
    -------
    MRI_image: torch.Tensor (N, H, W) 
        Batch of MRI images. N is the batch size, (H, W) is the image size.
    """
    assert mri_image.dim() == 3, "mri_image must be a 3D tensor"
    k_space = fft2(mri_image, dim=(1, 2))

    if apply_shift:
        k_space = fftshift(k_space, dim=(1, 2))

    return k_space

# apply soft thresholding
def soft_threshold(mri_image, threshold):
    """
    Apply soft thresholding
    -------
    input: torch.Tensor
        input tensor, stack of (N, H, W) MRI images
    threshold: float
        threshold value
    """
    assert mri_image.dim() == 3, "mri_image must be a 3D tensor"

    for idx, image in enumerate(mri_image):
        mri_image[idx] = torch.sign(image) * torch.max(torch.abs(image) - threshold, torch.zeros_like(image))

    return mri_image


# execute ISTA algorithm
def ISTA_MRI(mu, shrinkage, K, partial_k_space, M):
    """
    Execute ISTA algorithm
    -------
    mu: float
        step size mu
    shrinkage: float
        shrinkage value (lambda)
    K: int
        number of iterations
    partial_k_space: torch.Tensor (N, H, W)
        partial k-space of MRI image 
    M: torch.Tensor (N, H, W)
        k-space sampling mask 
    """
    # convert partial_k_space to MRI image
    y = kspace_to_mri(partial_k_space, reverse_shift=False)

    # initialize 
    x_t = y
    FY = partial_k_space

    # run ISTA
    for i in range(K):
        # (1) apply soft thresholding to x_t
        x_t = soft_threshold(x_t, shrinkage)

        # (2) data consistency step abs(Finv(FX - mu*M*FX + mu*FY))
        FX = mri_to_kspace(x_t, apply_shift=True)         
        MFX = apply_mask_k_space(FX, M)

        z = FX - mu*MFX + mu*FY
        x_t = kspace_to_mri(z, reverse_shift=False)

    # store resulting image
    return x_t


def plot_single_mri_image(mri_image, title, save_path):
    """
    Plot single MRI image
    -------
    mri_image: torch.Tensor (N, H, W)
        MRI image
    title: str
        title of plot
    save_path: str
        path to save image
    """
    # plot image
    plt.figure(figsize=(10, 10))
    plt.imshow(mri_image.numpy(), cmap="gray")
    plt.title(title)
    plt.savefig(save_path)
    # plt.close()


def plot_k_space(k_space, save_path):
    """	Plot k-space. """

    plt.figure(figsize = (10, 6))
    plt.imshow(make_plot_friendly(k_space), vmin=-2.3,interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title('k-space')
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')

def plot_ex4c(test_acc_mri, test_x_out, test_gt, save_path):

    plt.figure(figsize = (10, 6))
    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(test_acc_mri[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Accelerated MRI')

        plt.subplot(3,5,i+6)
        plt.imshow(test_x_out[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('ISTA reconstruction')

        plt.subplot(3,5,i+11)
        plt.imshow(test_gt[i+1,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Ground-truth MRI')

    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.show()

def make_plot_friendly(input_image):
    """	Increase dynamic range of the k-space image """
    return torch.log(torch.abs(input_image) + 1e-20)

def calculate_mse(input_image, ground_truth):
    """	Calculate mean squared error """
    return torch.mean((input_image - ground_truth)**2)


# calculate validation loss for ISTA algorithm
def calculate_loss_ista(data_loader, criterion, mu, shrinkage, K, apply_ista=True):
    """
    Calculate the loss on the given data set.
    -------
    data_loader : torch.utils.data.DataLoader
        Data loader to use for the data set.
    criterion : torch.nn.modules.loss
        Loss function to use.
    device : torch.device
        Device to use for the model.
    -------
    loss : float    
        The loss on the data set.
    """

    # initialize loss
    loss = 0

    # loop over batches
    for i, (partial_kspace, M, gt_mri) in enumerate(tqdm(data_loader, position=0, leave=False, ascii=False)):

            # forward pass
            if apply_ista:
                ista_mri = ISTA_MRI(mu, shrinkage, K, partial_kspace, M)
            else:
                ista_mri = kspace_to_mri(partial_kspace, reverse_shift=False)
           
            # calculate loss
            loss += criterion(ista_mri, gt_mri)

    # return the loss
    return loss / len(data_loader)

if __name__ == "__main__":
    # parameters
    mu = 0.8
    shrinkage = 0.15
    K = 30

    data_loc = 'assignment_4/Fast_MRI_Knee/' #change the datalocation to something that works for you
    batch_size = 6

    # create dataloaders
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    # for i, (partial_kspace, M, gt_mri) in enumerate(test_loader):
    #     if i == 1:
    #         break

    #### exc 4c ####
    # apply ISTA algorithm to MRI images
    # ista_mri = ISTA_MRI(mu, shrinkage, K, partial_kspace, M)
    # accel_mri = kspace_to_mri(partial_kspace, reverse_shift=True)

    # plot_ex4c(accel_mri, ista_mri, gt_mri,'assignment_4/figures/exc_4c.png')

    #### exc 4d ####
    # calculate mse on accelerated MRI images
    mse = torch.nn.MSELoss()
    mse_loss_accelerated_MRI = calculate_loss_ista(test_loader, mse, mu, shrinkage, K, apply_ista=False)
    print(f"MSE loss on accelerated MRI images: {mse_loss_accelerated_MRI}")

    # calculate mse on output of ISTA
    mse_loss_ista = calculate_loss_ista(test_loader, mse, mu, shrinkage, K, apply_ista=True)
    print(f"MSE loss ISTA out: {mse_loss_ista}") 