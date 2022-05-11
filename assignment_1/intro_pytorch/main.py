from MNIST_dataloader import create_dataloaders
from model import build_model
import matplotlib.pyplot as plt
import torch
import numpy as np

from train import train_model, calculate_loss
from model import load_model, relu

# Plot two graphs side-by-side, 1) Overlay the prediction from the untrained model and
# the trained model, 2) The test set data. As usual, make sure to have the proper labels
# and legends
def plot_question_1f(clean_images, noisy_images, pred_untrained, pred_trained, num_examples=10):
    """
    Plots the first 10 images from the test set, overlayed with the prediction from the
    untrained model and the trained model.
    -------
    clean_images: torch.Tensor
        The clean images
    noisy_images: torch.Tensor
        The noisy images
    pred_untrained: torch.Tensor
        The prediction from the untrained model
    pred_trained: torch.Tensor
        The prediction from the trained model
    num_examples: int
        The number of examples to plot
    """
    # show the examples in a plot
    plt.figure(figsize=(12, 3))
    n_rows = 4

    for i in range(num_examples):
        plt.subplot(n_rows, num_examples, i+1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(n_rows, num_examples, i + num_examples + 1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(n_rows, num_examples, i + 2*num_examples + 1)
        plt.imshow(pred_untrained[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(n_rows, num_examples, i + 3*num_examples + 1)
        plt.imshow(pred_trained[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig("question_1f.png", dpi=300, bbox_inches='tight')
    plt.show()



def plot_examples(clean_images, noisy_images, prediction, num_examples=10):
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
        plt.subplot(3, num_examples, i+1)
        plt.imshow(clean_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + num_examples + 1)
        plt.imshow(noisy_images[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, num_examples, i + 2*num_examples + 1)
        plt.imshow(prediction[i, 0, :, :], cmap='gray')
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

def plot_loss(train_losses, valid_losses):
    """
    Plots the loss.
    -------
    train_losses: list
        The training loss
    valid_losses: list
        The validation loss
    """
    num_epochs = len(train_losses)

    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_losses, label='Training loss')
    ax.plot(valid_losses, label='Validation loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch[n]', fontsize="x-large")
    plt.ylabel('Loss', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.xticks(np.arange(0, num_epochs, 10))
    plt.savefig("loss.png", dpi=300, bbox_inches='tight')
    plt.show()

def relu_plot_2a(x):
    """
    Plots the ReLU function.
    -------
    x: torch.Tensor
        The input to the ReLU function
    """
    # plot the ReLU function
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, relu(x), linewidth=4)
    ax.set_xlim(-9, 10)
    ax.set_ylim(-1, 10)
    plt.xlabel('x', fontsize="x-large")
    plt.ylabel('y', fontsize="x-large")
    plt.xticks(np.arange(-9, 10, 1))
    plt.grid(True)
    plt.savefig("relu.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # define parameters
    data_loc = 'assignment_1/intro_pytorch/data' # change the datalocation to something that works for you
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 50
    
    # get dataloader
    train_loader, valid_loader, test_loader = create_dataloaders(data_loc, batch_size)

    # print dataset lengths
    print("Train set length:", len(train_loader.dataset))
    print("Valid set length:", len(valid_loader.dataset))
    print("Test set length:", len(test_loader.dataset))   

    # plot some examples
    # plot_examples(test_loader.dataset.Noisy_Images, test_loader.dataset.Clean_Images, None)    

    # define the model
    model = build_model()

    # define the loss function
    criterion = torch.nn.MSELoss()

    # use adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # RMSprop optimizer
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-2)

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
    untrained_prediction = model(x_noisy_example)

    # # move back to cpu    
    untrained_prediction = untrained_prediction.detach().cpu()
    x_noisy_example = x_noisy_example.detach().cpu()

    # # convert back to 32x32 images
    untrained_prediction = reshape_images(untrained_prediction)

    # # # plot the prediction next to the original image
    # plot_examples(x_clean_example, x_noisy_example, untrained_prediction)     

    # # train the model 
    model, train_losses, valid_losses = train_model(model, 
                                                    train_loader, valid_loader, 
                                                    optimizer, criterion, num_epochs, 
                                                    device, write_to_file=True)
     
    # ##### do a prediction on trained model #####    
    # load the trained model 
    # model = load_model(model, "model_params.pth")
    trained_prediction = model(x_noisy_example.to(device))
    trained_prediction = trained_prediction.detach().cpu()
    trained_prediction = reshape_images(trained_prediction)

    # # question 1f plot
    # plot_question_1f(x_clean_example, x_noisy_example, untrained_prediction, trained_prediction)

    plot_examples(untrained_prediction, trained_prediction, x_clean_example)

    # plot_examples(x_clean_example, x_noisy_example.detach().cpu(), reshape_images(trained_prediction.detach().cpu()))  

    # plot the loss
    plot_loss(train_losses, valid_losses)

    # calculate loss on test set
    test_loss = calculate_loss(model, test_loader, criterion, device)
    print("Test set loss:", test_loss)

    ### excercise 2 #####
    # plot ReLU
    # relu_plot_2a(torch.linspace(-9, 10, 20))
    


# if the file is run as a script, run the main function
if __name__ == '__main__':
    main()
