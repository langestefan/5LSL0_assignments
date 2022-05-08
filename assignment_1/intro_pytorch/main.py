from MNIST_dataloader import create_dataloaders
from model import build_model
import matplotlib.pyplot as plt
import torch
import numpy as np

from train import train_model
from model import load_model

def plot_examples(noisy_images, clean_images, num_examples=10):
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
        plt.subplot(2, num_examples, i+1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2, num_examples, i + num_examples + 1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("data_examples.png", dpi=300, bbox_inches='tight')
    plt.show()

def reshape_images(images):
    """
    Reshapes the images to be 32x32x1
    -------
    images: torch.Tensor
        The images to reshape
    """
    # reshape the images to be 32x32x1
    images_reshaped = torch.reshape(images, (images.shape[0], 1, 32, 32))
    return images_reshaped


def main():
    # define parameters
    data_loc = 'assignment_1/intro_pytorch/data' # change the datalocation to something that works for you
    batch_size = 32
    learning_rate = 0.1
    num_epochs = 50
    
    # get dataloader
    train_loader, valid_loader, test_loader = create_dataloaders(data_loc, batch_size)

    # print dataset lengths
    print("Train set length:", len(train_loader.dataset))
    print("Valid set length:", len(valid_loader.dataset))
    print("Test set length:", len(test_loader.dataset))   

    # plot some examples
    # plot_examples(test_loader.dataset.Noisy_Images, test_loader.dataset.Clean_Images)

    # define the model
    model = build_model()

    # define the loss function
    criterion = torch.nn.MSELoss()

    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # move model to device
    model.to(device)

    ##### do a prediction on untrained model #####
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    x_noisy_example = x_noisy_example.to(device)

    # # get the prediction
    # prediction = model(x_noisy_example)

    # # move back to cpu    
    # prediction = prediction.detach().cpu()
    # x_noisy_example = x_noisy_example.detach().cpu()

    # # convert back to 32x32 images
    # prediction = reshape_images(prediction)

    # # plot the prediction next to the original image
    # # plot_examples(x_noisy_example, prediction)     

    # load the trained model 
    # model = load_model(model, "model_params.pth")

    # train the model 
    model, train_losses, valid_losses = train_model(model, 
                                                    train_loader, test_loader, #valid_loader, 
                                                    optimizer, criterion, num_epochs, 
                                                    device, write_to_file=True)
     
    ##### do a prediction on trained model #####
    prediction = model(x_noisy_example)
    plot_examples(x_noisy_example.detach().cpu(), reshape_images(prediction.detach().cpu()))  

    # plot the loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    # axis labels
    plt.xlabel('Epoch[n]')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 1))
    plt.savefig("loss.png", dpi=300, bbox_inches='tight')
    plt.show()


# if the file is run as a script, run the main function
if __name__ == '__main__':
    main()
