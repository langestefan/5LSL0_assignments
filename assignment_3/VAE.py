# imports
from audioop import lin2adpcm
from sklearn import linear_model
import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        # create layers here
        self.encoder = nn.Sequential(
            # input is N, 1, 32, 32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=1), # N, 16, 2, 2
            nn.Flatten(),
            # Linear layer to output the mean and the std of the latent space
        )   
        self.x_mean = nn.Linear(in_features=64, out_features=2)
        self.x_std = nn.Linear(in_features=64, out_features=2)
        
    def sampling (self,x_mean,log_x_std):
        # sampling from the latent space
        epsilon = torch.randn_like(x_mean)
        x_std = torch.exp(0.5*log_x_std)
        x_sample = x_mean + x_std * epsilon
        return x_sample

    def forward(self, x):
        # forward pass through the encoder
        h = self.encoder(x)
        x_mean = self.x_mean(h)
        x_log_var = self.x_std(h)
        x_sample = self.sampling(x_mean, x_log_var)

        return x_sample, x_mean, x_log_var
    
# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #input is 1,2
        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=64), # 1,16
            Reshape(-1,16,2,2), # 1, 16, 2, 2
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=2, padding=1), # N, 1, 32, 32
            nn.Sigmoid(),
        )
        
    def forward(self, h):
        # use the created layers here
        r = self.decoder(h)
        return r
    
# VAutoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x_sample, x_mean, x_log_var = self.encoder(x)
        output_encoder = self.decoder(x_sample)
        return output_encoder, x_sample, x_mean, x_log_var



def test():
    # test encoder part 
    # x = torch.randn((1, 1, 32, 32))   
    # model = Encoder()

    # x_sample, z_mean, z_log_var = model(x)
    # print ('x_sample:', x_sample)
    # print ('z_mean:', z_mean)
    # print ('z_log_var:', z_log_var)


    # test decoder part
    # x = torch.randn((1, 2))
    # model = Decoder()

    # preds= model(x)
    # print(preds.shape)
    # print(x.shape)


    # test VAE
    x = torch.randn((64, 1, 32, 32))   
    model = VAE()

    output_encoder, x_sample, x_mean, x_log_varpreds = model(x)
    print ('x_input:', x.shape)
    print('output_encoder', output_encoder.shape)
    print('x_sample', x_sample.shape)
    print('x_mean', x_mean.shape)
    print('x_log_var', x_log_varpreds.shape)


if __name__ == "__main__":
    test()    