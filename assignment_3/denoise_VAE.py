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
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=1), # N, 16, 2, 2
            nn.BatchNorm2d(16),
            nn.Flatten(),
            # Linear layer to output the mean and the std of the latent space
        ) 
        self.x_mean = nn.Sequential(
            nn.Linear(64,16),
            #nn.ReLU()
        )  
        self.x_std = nn.Sequential(
            nn.Linear(64,16),
            #nn.ReLU()
        ) 
        self.n = torch.distributions.Normal(0,1)
        self.n.loc = self.n.loc.cuda()
        self.n.scale = self.n.scale.cuda()
        
    def sampling (self,x_mean,x_std):
        # sampling from the latent space
        epsilon = torch.distributions.Normal(0,1)
        #epsilon = epsilon.loc.cuda()
        #epsilon = epsilon.scale.cuda()
        epsilon = self.n.sample((x_mean.shape))
        x_sample = x_mean + 0.5*x_std * epsilon
        return x_sample

    def forward(self, x):
        # forward pass through the encoder
        h = self.encoder(x)
        x_mean = self.x_mean(h)
        x_std = torch.exp(self.x_std(h))
        x_sample = self.sampling(x_mean, x_std)

        return x_sample, x_mean, x_std
    
# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #input is 1,16
        self.decoder = nn.Sequential(
            nn.Linear(in_features=16, out_features=64), # 1,16
            Reshape(-1,16,2,2), # 1, 16, 2, 2
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(4, 4), stride=2, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=2, padding=1), # N, 1, 32, 32
            nn.BatchNorm2d(1),
            
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
        output_decoder = self.decoder(x_sample)
        return output_decoder, x_sample, x_mean, x_log_var



def test():
    # test encoder part 
    # x = torch.randn((1, 1, 32, 32))   
    # model = Encoder()

    # x_sample, z_mean, z_log_var = model(x)
    # print ('x_sample:', x_sample.shape)
    # print ('z_mean:', z_mean)
    # print ('z_log_var:', z_log_var)


    #test decoder part
    # x = torch.randn((1, 16))
    # model = Decoder()

    # preds= model(x)
    # print(preds.shape)
    # print(x.shape)


    # test VAE
    x = torch.randn((64, 1, 32, 32)).cuda()   
    model = VAE()

    output_decoder, x_sample, x_mean, x_log_varpreds = model(x).to(torch.device('cuda:0'))
    print ('x_input:', x.shape)
    print('output_decode', output_decoder.shape)
    print('x_sample', x_sample.shape)
    print('x_mean', x_mean.shape)
    print('x_log_var', x_log_varpreds.shape)


if __name__ == "__main__":
    test()    