# %% imports
import torch.optim as optim
import torch
import torch.nn as nn

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # create layers here
        self.encoder = nn.Sequential(
            # input is N, 1, 32, 32
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = 'same'), # N, 16, 32, 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # N, 16, 16, 16
            nn.Conv2d(in_channels =16, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = 'same'), # N, 16, 16, 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # N, 16, 8, 8
            nn.Conv2d(in_channels =16, out_channels = 16,kernel_size = (3, 3), stride = 1, padding = 'same'), # N, 16, 8, 8
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # N, 16, 4, 4
            nn.Conv2d(in_channels =16, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = 'same'), # N, 16, 4, 4
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # N, 16, 2, 2
            nn.Conv2d(in_channels =16, out_channels = 1, kernel_size = (2, 2), stride = 1, padding = 'same'), # N, 1, 2, 2
            nn.ReLU(True),
            nn.MaxPool2d((1,2), stride=1), # N, 1, 2, 1

        )
    def forward(self, x):
        # use the created layers here
        h = self.encoder(x)
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #input is N, 1, 2, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1, out_channels = 16, kernel_size = (2, 2), stride = 1, padding = 'same'), # N, 16, 2, 2
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor = 2), # N, 16, 4, 4
            nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 'same'), # N, 32, 20, 20
            #nn.ReLU(True),
            #nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 'same'), # N, 32, 40, 40
            #nn.ReLU(True),
            #nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 'same'), # N, 16, 80, 80
            #nn.ReLU(True),
            #nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 2, padding = 'same'), # N, 1, 160, 240
            #nn.Tanh()
        )
        
    def forward(self, h):
        # use the created layers here
        r = self.decoder(h)
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

def test():
    # test encoder part 
    #x = torch.randn((1, 1, 32, 32))   
    #model = Encoder()
    
    # test decoder part
    x = torch.randn((1, 1, 2, 1))
    model = Decoder()

    preds = model(x)
    print ('preds :',preds.shape)
    print ('x :',x.shape)

if __name__ == "__main__":
    test()    
