import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class Gen_h2l(nn.Module):
    # 64*64--->16*16
    def __init__(self,numIn=3, numOut=3,add_n = False,fea=False):
        super(Gen_h2l,self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.add_n = add_n
        self.fea = fea
        if add_n:
            #self.noise = Variable(torch.randn(1,64).cuda())
            self.linear = nn.Linear(64,64*64)
            self.conv1 = nn.Conv2d(3+1,64,kernel_size=3,stride=1,padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.acti1 = nn.ReLU(True)
        conv2_1 = []
        bn2_1 = []
        relu2 = []
        conv2_2 = []
        bn2_2 = []
        for idx in range(12):
            conv2_1.append(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1))
            bn2_1.append(nn.BatchNorm2d(64))
            relu2.append(nn.ReLU(True))
            conv2_2.append(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1))
            bn2_2.append(nn.BatchNorm2d(64))

        self.conv2_1 = nn.ModuleList(conv2_1)
        self.bn2_1 = nn.ModuleList(bn2_1)
        self.relu2 = nn.ModuleList(relu2)
        self.conv2_2 = nn.ModuleList(conv2_2)
        self.bn2_2 = nn.ModuleList(bn2_2)

        self.bn3 = nn.BatchNorm2d(64)
        self.acti2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(64, 64,kernel_size=3,stride=2,padding=1)

        conv3_1 = []
        bn3_1 = []
        relu3 = []
        conv3_2 = []
        bn3_2 = []
        for idx in range(3):
            conv3_1.append(nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1))
            bn3_1.append(nn.BatchNorm2d(64))
            relu3.append(nn.ReLU(True))
            conv3_2.append(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1))
            bn3_2.append(nn.BatchNorm2d(64))

        self.conv3_1 = nn.ModuleList(conv3_1)
        self.bn3_1 = nn.ModuleList(bn3_1)
        self.relu3 = nn.ModuleList(relu3)
        self.conv3_2 = nn.ModuleList(conv3_2)
        self.bn3_2 = nn.ModuleList(bn3_2)


        self.bn4 = nn.BatchNorm2d(64)
        self.acti3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2,padding=1)

        conv4_1 = []
        relu4 = []
        for idx in range(2):
            conv4_1.append(nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1))
            relu4.append(nn.ReLU())
        self.conv4_1 = nn.ModuleList(conv4_1)
        self.relu4 = nn.ModuleList(relu4)
        self.conv5 = nn.Conv2d(64, numOut,kernel_size=3,stride=1,padding=1)
        #self.acti4 = nn.Sigmoid()
        self.acti4 = nn.Tanh()

    def forward(self, x,noise=None):
        if self.add_n:
            noise = self.linear(noise)
            noise= noise.view(x.size(0),1,x.size(2),x.size(3))
            x = torch.cat((x,noise),1)

        out = self.conv1(x)
        out = self.acti1(out)

        for idx in range(12):
            out1 = self.conv2_1[idx](out)
            out1 = self.bn2_1[idx](out1)
            out1 = self.relu2[idx](out1)
            out1 = self.conv2_2[idx](out1)
            out1 = self.bn2_2[idx](out1)
            out = out1+out

        out = self.bn3(out)
        out = self.acti2(out)
        out = self.conv3(out)

        for idx in range(3):
            out1 = self.conv3_1[idx](out)
            out1 = self.bn3_1[idx](out1)
            out1 = self.relu3[idx](out1)
            out1 = self.conv3_2[idx](out1)
            out1 = self.bn3_2[idx](out1)
            out = out1 + out

        out = self.bn4(out)
        out = self.acti3(out)
        out = self.conv4(out)
        if self.fea:
            fea_extract=out #64*16*16
        for idx in range(2):
            out1 = self.conv4_1[idx](out)
            out1 = self.relu4[idx](out1)
            out = out + out1

        out = self.conv5(out)
        out = self.acti4(out)
        if self.fea:
            return fea_extract,out
        else:
            return out

class Gen_h2l_2(nn.Module):
    # 64*64--->16*16
    def __init__(self,numIn=3, numOut=3,add_n=False,fea=False):
        super(Gen_h2l_2,self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.add_n = add_n
        self.fea = fea
        nf = 64
        if add_n:
            self.linear = nn.Linear(64,64*64)
            self.conv1 = nn.Conv2d(numIn+1,nf,kernel_size=3,stride=2,padding=1)#32
        else:
            self.conv1 = nn.Conv2d(numIn,nf, kernel_size=3, stride=2, padding=1)#32

        self.main = []
        for idx in range(2):
            self.main += [BasicBlock(nf,nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#16

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#8

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#4
        self.main = nn.Sequential(*self.main)

        self.conv2 = nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(nf,numOut,kernel_size=3,stride=1,padding=1)
        self.acti = nn.Tanh()

    def forward(self,x,noise=None):
        if self.add_n:
            noise = self.linear(noise)
            noise= noise.view(x.size(0),1,x.size(2),x.size(3))
            x = torch.cat((x,noise),1)

        out = F.relu(self.conv1(x))
        fea = self.main(out)
        out = self.conv2(fea)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv3(out)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv4(out)
        out = self.acti(out)
        if self.fea:
            return fea,out
        else:
            return out

class Gen_h2l_cgan(nn.Module):
    # 64*64--->16*16
    def __init__(self,numIn=3, numOut=3,add_n=False,fea=False):
        super(Gen_h2l_cgan,self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.add_n = add_n
        self.fea = fea
        nf = 64
        #2 res_d-2 res d-2-res -2
        if add_n:
            self.conv1 = nn.Conv2d(numIn+8,nf,kernel_size=3,stride=2,padding=1)#32
        else:
            self.conv1 = nn.Conv2d(numIn,nf, kernel_size=3, stride=2, padding=1)#32

        self.main = []
        for idx in range(2):
            self.main += [BasicBlock(nf,nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#16

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#8

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#4
        self.main = nn.Sequential(*self.main)

        self.conv2 = nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(nf,numOut,kernel_size=3,stride=1,padding=1)
        self.acti = nn.Tanh()

    def forward(self, x,noise=None):
        if self.add_n:
            noise= noise.view(x.size(0),noise.size(1),1,1).expand(noise.size(0),noise.size(1),x.size(2),x.size(3))
            x = torch.cat((x,noise),1)

        out = F.relu(self.conv1(x))
        fea = self.main(out)
        out = self.conv2(fea)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv3(out)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv4(out)
        out = self.acti(out)
        if self.fea:
            return fea,out
        else:
            return out

#!!!!!!!!!!!!!
class Gen_h2l_0424(nn.Module):
    # 64*64--->16*16
    def __init__(self, numIn=3, numOut=3,lv=64,add_n=False,fea=False):
        super(Gen_h2l_0424, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.add_n = add_n
        self.fea = fea
        nf = 64
        #2 res_d-2 res d-2-res -2
        # if add_n:
        #     self.linear = nn.Linear(lv, 64 * 64)
        #     self.conv1_ = nn.Conv2d(numIn+1,nf,kernel_size=3,stride=1,padding=1)
        # else:
        #     self.conv1_ = nn.Conv2d(numIn,nf, kernel_size=3, stride=1, padding=1)
        if add_n:
            self.weight = nn.Parameter(torch.zeros(numIn))
        self.conv1 = nn.Conv2d(numIn, nf, kernel_size=3, stride=1, padding=1)
        self.main = []
        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)]  #32

        for idx in range(2):
            self.main += [BasicBlock(nf,nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#16

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#8

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#4

        self.main = nn.Sequential(*self.main)

        self.up1 = []
        for idx in range(2):
            self.up1 += [BasicBlock(nf, nf)]
        self.up1 += [nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)]
        self.up1 = nn.Sequential(*self.up1)

        self.up2 = []
        for idx in range(2):
            self.up2 += [BasicBlock(nf, nf)]
        self.up2 += [nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)]
        self.up2 = nn.Sequential(*self.up2)

        self.conv4 = nn.Conv2d(nf,numOut,kernel_size=3,stride=1,padding=1)
        self.acti = nn.Tanh()

    def forward(self, x):
        # if self.add_n:
        #     noise = self.weight(noise)
        #     noise = noise.view(x.size(0),1,x.size(2),x.size(3))
        #     x = torch.cat((x,noise),1)
        if self.add_n:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = x + self.weight.view(1, -1, 1, 1) * noise
        out = F.relu(self.conv1(x))
        fea = self.main(out)
        out = self.up1(fea)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.up2(out)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv4(out)
        out = self.acti(out)
        if self.fea:
            return fea,out
        else:
            return out

class Gen_h2l_0424_32(nn.Module):
    # 64*64--->16*16
    def __init__(self,numIn=3, numOut=3,lv=64,add_n=False,fea=False):
        super(Gen_h2l_0424_32,self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.add_n = add_n
        self.fea = fea
        nf = 64
        #2 res_d-2 res d-2-res -2
        if add_n:
            self.linear = nn.Linear(lv, 64 * 64)
            self.conv1 = nn.Conv2d(numIn+1,nf,kernel_size=3,stride=1,padding=1)
        else:
            self.conv1 = nn.Conv2d(numIn,nf, kernel_size=3, stride=1, padding=1)#64

        self.main = []
        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)]#32

        for idx in range(2):
            self.main += [BasicBlock(nf,nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main += [nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#16

        for idx in range(2):
            self.main += [BasicBlock(nf, nf)]

        self.main += [nn.BatchNorm2d(nf)]
        self.main += [nn.ReLU(True)]
        self.main +=[nn.Conv2d(nf, nf,kernel_size=3,stride=2,padding=1)]#8
        self.main = nn.Sequential(*self.main)

        self.up1 = []
        for idx in range(2):
            self.up1 += [BasicBlock(nf, nf)]
        self.up1 += [nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)]#16
        self.up1 = nn.Sequential(*self.up1)

        self.up2 = []
        for idx in range(2):
            self.up2 += [BasicBlock(nf, nf)]
        self.up2 += [nn.Conv2d(nf, nf*4, kernel_size=3, stride=1, padding=1)]#32
        self.up2 = nn.Sequential(*self.up2)

        self.conv4 = nn.Conv2d(nf,numOut,kernel_size=3,stride=1,padding=1)#32
        self.acti = nn.Tanh()

    def forward(self, x,noise=None):
        if self.add_n:
            noise = self.linear(noise)
            noise = noise.view(x.size(0),1,x.size(2),x.size(3))
            x = torch.cat((x,noise),1)
        out = F.relu(self.conv1(x))
        fea = self.main(out)
        out = self.up1(fea)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.up2(out)
        out = F.relu(F.pixel_shuffle(out, 2))
        out = self.conv4(out)
        out = self.acti(out)
        if self.fea:
            return fea,out
        else:
            return out

