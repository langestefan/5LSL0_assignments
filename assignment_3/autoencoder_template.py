# %% imports
import torch
import torch.nn as nn

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # create layers here
        
    def forward(self, x):
        # use the created layers here
        h = x
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
        
    def forward(self, h):
        # use the created layers here
        r = h
        return r
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    
