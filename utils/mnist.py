import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
    
class I_MNIST(nn.Module):
    def __init__(self, nz=5):
        super(I_MNIST, self).__init__()
        # Load pre-trained ResNet18
        # self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = models.resnet18()
        # self.resnet = models.resnet34()
        
        # Modify the first convolutional layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(1, self.resnet.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify the final fully connected layer to output nz features
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, nz)
        
        # Remove the initial maxpool layer as FashionMNIST images are smaller
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Reshape the input
        x = x.view(-1, 1, 28, 28)
        return self.resnet(x)


class I_MNIST2(models.ResNet):
    def __init__(self, nz=5):
        # Initialize with the basic block and layer configuration of ResNet-34
        super(I_MNIST2, self).__init__(block=models.resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=nz)
        self.conv1 = nn.Conv2d(1, self.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace the maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Reshape the input
        x = x.view(-1, 1, 28, 28)
        # Call the original forward method
        return super(I_MNIST2, self).forward(x)

class I_MNIST3(nn.Module):
    def __init__(self, nz=5):
        super(I_MNIST3, self).__init__()
        # Load the pre-trained VGG16 model
        original_vgg16 = models.vgg16()

        # Modify the first convolutional layer to accept 1 channel instead of 3
        original_vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Since MNIST images are 28x28, which is smaller than what VGG16 expects,
        # we need to remove some of the later layers that reduce the image size too much.
        # Remove the last max pooling layer to adapt to the smaller input size
        original_vgg16.features = nn.Sequential(*list(original_vgg16.features.children())[:-1])

        # Adapt the classifier
        # The input features to the classifier need to be adjusted.
        # This number depends on the output size of the last convolutional layer.
        # Assuming the output size is 7x7, we calculate 7*7*512
        original_vgg16.classifier[0] = nn.Linear(7*7*512, 4096)

        # Replace the final layer in the classifier to output 'nz' classes
        original_vgg16.classifier[6] = nn.Linear(4096, nz)

        self.model = original_vgg16

    def forward(self, x):
        # Reshape the input to match the expected format of VGG16
        # The original VGG16 expects 3-channel images, but we've adapted it to 1-channel
        x = x.view(-1, 1, 28, 28)
        return self.model(x)


    
class G_MNIST(nn.Module):
    def __init__(self, nz, nc=1, ngf=32):
        super(G_MNIST, self).__init__()
        self.nz = nz
        layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) 
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]
        if nc == 3:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ])
        else:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=False),
                nn.Tanh()
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        output = self.main(input)
        return output.view(-1, 28*28)

    

class f_MNIST(nn.Module):
    def __init__(self, nz, ndf = 16, power = 6):
        super(f_MNIST, self).__init__()
        self.power = power
        layers = [
            # input is (nz) 
            # state size. (ndf * 4) 
            nn.Linear(nz * power , ndf * 4),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf * 2) 
            nn.Linear(ndf * 4, ndf * 2),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf) 
            nn.Linear(ndf * 2, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        # dist = nn.PairwiseDistance(input, p=2)
        powers = [i for i in range(0, self.power)]
        input = torch.cat([torch.pow(input, i) for i in powers], dim=1)
        output = self.main(input)
        return output.squeeze(1)
    