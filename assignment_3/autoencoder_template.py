# imports
import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # create layers here
        self.encoder = nn.Sequential(
            # input is N, 1, 32, 32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2), # N, 16, 16, 16
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2), # N, 16, 8, 8
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2), # N, 16, 4, 4
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2), # N, 16, 2, 2
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3,3), stride=1, padding=(2,1)), # N, 1, 4, 2
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool2d(2), # N, 1, 2, 1

        )
    def forward(self, x):
        # use the created layers here
        h = self.encoder(x)
        return h
    
# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #input is N, 1, 2, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=1, padding=0), # N, 16, 2, 1
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(1,2), mode='nearest'), # N, 16, 2, 2
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 2, 2
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'), # N, 16, 4, 4
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'), # N, 16, 8, 8
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'), # N, 16, 16, 16
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 1, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2,2), mode='nearest'), # N, 16, 32, 32
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1, padding=1), # N, 1, 32, 32
            #nn.BatchNorm2d(1),
            #nn.ReLU(True)
        )
        
    def forward(self, h):
        # use the created layers here
        r = self.decoder(h)
        return r
    
# Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h


# classifier for excercise 4
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # create layers here
        self.classifier = nn.Sequential(
            # input is N, 1, 32, 32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 32, 32
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # N, 16, 16, 16
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # N, 16, 8, 8
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 8, 8
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # N, 16, 4, 4
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1), # N, 16, 4, 4
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # N, 16, 2, 2 = 64

            # classification head
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10))
            # we will use CrossEntropyLoss, which already uses a LogSoftmax # nn.LogSoftmax(dim=1)

        
    def forward(self, x):
        # use the created layers here
        classifier_scores = self.classifier(x)
        return classifier_scores


def test():
    # test encoder part 
    #x = torch.randn((1, 1, 32, 32))   
    #model = Encoder()
    
    # test decoder part
    #x = torch.randn((1, 1, 2, 1))
    #model = Decoder()

    # test AE
    x = torch.randn((64, 1, 32, 32))   
    model = AE()

    preds, latent = model(x)
    #preds = model(x)
    print('latent:', latent.shape)
    print ('preds :',preds.shape)
    print ('x :',x.shape)

if __name__ == "__main__":
    test()    
