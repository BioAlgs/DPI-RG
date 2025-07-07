import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class I_CIFAR10(models.ResNet):
    def __init__(self, nz=5, img_size=32):
        # Initialize with the basic block and layer configuration of ResNet-18
        super(I_CIFAR10, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=nz)
        self.conv1 = nn.Conv2d(3, self.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_size = img_size
    def forward(self, x):
        x = x.view(-1, 3, self.img_size, self.img_size)
        return super(I_CIFAR10, self).forward(x)


class G_CIFAR10(nn.Module):
    def __init__(self, nz, nc=3, ngf=16, img_size=32):
        super(G_CIFAR10, self).__init__()
        self.img_size = img_size
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
        if img_size == 32:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ])
        elif img_size == 64:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        output = self.main(input)
        return output.view(-1, 3, self.img_size**2)


    
class f_CIFAR10(nn.Module):
    def __init__(self, nz, ndf = 16, power = 6):
        super(f_CIFAR10, self).__init__()
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
        # print(input.shape)
        # dist = nn.PairwiseDistance(input, p=2)
        powers = [i for i in range(0, self.power)]
        input = torch.cat([torch.pow(input, i) for i in powers], dim=1)
        # print(powers.shape)
        output = self.main(input)
        return output.squeeze(1)