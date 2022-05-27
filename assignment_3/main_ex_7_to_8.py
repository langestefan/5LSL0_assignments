# libraries
from matplotlib import markers
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

# from tqdm import tqdm
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import autoencoder_template
import train

# nearest neighbor excercise 3
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

# to fix a bug with numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":
    # set torches random seed
    torch.random.manual_seed(0)