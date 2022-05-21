# %% imports
# libraries
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# %% MAP Estimation
# parameters
no_iterations = 1000
learning_rate = 1e-2
beta = 0.01

estimated_latent = nn.Parameter(torch.randn(10,16))
optimizer_map = torch.optim.Adam([estimated_latent],lr = learning_rate)

# optimization
for i in tqdm(range(no_iterations)):
    optimizer_map.zero_grad()
    
    loss = ...
    
    loss.backward()
    optimizer_map.step()