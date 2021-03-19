from torch import nn
import torch.nn.functional as F
import numpy as np
from .spectral_normalization import SpectralNorm

import torch
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        #print(self.model(x).size())
        #print(self.bypass(x).size())
        return self.model(x) + self.bypass(x)

channels=3
GEN_SIZE=128
DISC_SIZE=128

class Discriminator(nn.Module):
    def __init__(self):
        #16*16
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(4),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))

class Discriminator_32(nn.Module):
    def __init__(self):
        #32*32
        super(Discriminator_32, self).__init__()
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                nn.ReLU(),
                nn.AvgPool2d(2),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))

class Discriminator_8(nn.Module):
    # 8*8 discriminator
    def __init__(self):
        super(Discriminator_8, self).__init__()
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))

class Discriminator_16(nn.Module):
    def __init__(self,add_n=False):
        super(Discriminator_16, self).__init__()
        self.add_noise = add_n
        if self.add_noise:
            self.linear = nn.Linear(64,16*16)
            self.model = nn.Sequential(
                    FirstResBlockDiscriminator(channels+1, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    nn.ReLU(),
                    nn.AvgPool2d(4),
                )
        else:
            self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(4),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x,noise=None):
        if self.add_noise:
            noise = self.linear(noise)
            noise = noise.view(x.size(0), 1, x.size(2), x.size(3))
            x = torch.cat((x, noise), 1)
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))
    
class Discriminator_32_(nn.Module):
    def __init__(self,add_n=False):
        super(Discriminator_32_, self).__init__()
        self.add_noise = add_n
        if self.add_noise:
            self.linear = nn.Linear(64,16*16)
            self.model = nn.Sequential(
                    FirstResBlockDiscriminator(channels+1, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    nn.ReLU(),
                    nn.AvgPool2d(4),
                )
        else:
            self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(4),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x,noise=None):
        if self.add_noise:
            noise = self.linear(noise)
            noise = noise.view(x.size(0), 1, x.size(2), x.size(3))
            x = torch.cat((x, noise), 1)
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))

class Discriminator_4(nn.Module):
    def __init__(self):
        super(Discriminator_4, self).__init__()
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(64, DISC_SIZE, stride=2),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
            nn.ReLU()
        )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))

class Discriminator_16_c(nn.Module):
    def __init__(self,add_n=False):
        super(Discriminator_16_c, self).__init__()
        self.add_noise = add_n
        if add_n:
            self.model = nn.Sequential(
                    FirstResBlockDiscriminator(channels+8, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                    nn.ReLU(),
                    nn.AvgPool2d(2),
                )
        else:
            self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(2),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x,noise=None):
        if self.add_noise:
            noise = noise.view(noise.size(0),noise.size(1),1, 1).expand(noise.size(0),noise.size(1), x.size(2), x.size(3))
            x = torch.cat((x, noise), 1)
        out = self.model(x)
        return self.fc(out.view(-1,DISC_SIZE))


class Discriminator_3D(nn.Module):
    def __init__(self):
        super(Discriminator_3D, self).__init__()
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,stride=2),

                nn.ReLU(),
                nn.AvgPool2d(7),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        out = self.model(x)
        #print(out.size())
        #exit()
        return self.fc(out.view(-1,DISC_SIZE))



#net = Discriminator_ms().cuda()
#output = net(torch.ones(1,3,64,64).cuda())
#print(len(output))