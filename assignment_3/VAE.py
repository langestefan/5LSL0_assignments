# imports
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
        self.conv = nn.Sequential(
            # input is N, 1, 32, 32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=1), # N, 16, 2, 2
            nn.BatchNorm2d(16),
            nn.Flatten()
            # Linear layer to output the mean and the std of the latent space
        ) 

        # mean of latent distribution
        self.x_mean = nn.Sequential(
            nn.Linear(64, 2),
            # nn.ReLU()
        )  

        # log variance of latent distribution
        self.x_log_stddev = nn.Sequential(
            nn.Linear(64, 2),
            # nn.ReLU()
        ) 

        # normal distribution
        self.normal = torch.distributions.Normal(0, 1)


    def reparameterize(self, mu, log_stddev):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        # std = exp(log_var / 2)
        std = torch.exp(log_stddev + 1e-7) # standard deviation

        # print("mu shape: ", mu.shape)

        # sampling epsilon from the N(0, 1) distribution
        eps = self.normal.sample(mu.shape).to(mu.device)  # `randn_like` as we need the same size
        
        sample = mu + eps*std # sampling
        return sample        

    def forward(self, x):
        # forward pass through the first part of the encoder
        hidden = self.conv(x) 
        
        # then get the mean and std from the convolutional layer output
        x_mean = self.x_mean(hidden)
        x_log_stddev = self.x_log_stddev(hidden)

        # get x_sample from the sampling function
        x_sample = self.reparameterize(x_mean, x_log_stddev)

        return x_sample, x_mean, x_log_stddev
    

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #input is 1,2
        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=64), # 1,16
            Reshape(-1,16,2,2), # 1, 16, 2, 2
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=2, padding=1), # N, 1, 32, 32
            # nn.BatchNorm2d(1),
            
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
        x_sample, x_mean, x_log_stddev = self.encoder(x)
        output_decoder = self.decoder(x_sample)
        return output_decoder, x_sample, x_mean, x_log_stddev



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

    output_decoder, x_sample, x_mean, x_log_varpreds = model(x)
    print ('x_input:', x.shape)
    print('output_decode', output_decoder.shape)
    print('x_sample', x_sample.shape)
    print('x_mean', x_mean.shape)
    print('x_log_var', x_log_varpreds.shape)


if __name__ == "__main__":
    test()    