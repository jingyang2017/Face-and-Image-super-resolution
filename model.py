import torch.nn as nn
import torch
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, upsample=False, nobn = False):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.nobn = nobn
        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv2 =nn.Sequential(nn.AvgPool2d(2,2), conv3x3(planes, planes))
        else:
            self.conv2 = conv3x3(planes, planes)
        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes or self.upsample or self.downsample:
            if self.upsample:
                self.skip = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
            elif self.downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2,2), nn.Conv2d(inplanes, planes, 1, 1))
            else:
                self.skip = nn.Conv2d(inplanes, planes, 1, 1, 0)
        else:
            self.skip = None
        self.stride = stride

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out


class GEN_DEEP(nn.Module):
    def __init__(self, ngpu=1):
        super(GEN_DEEP, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 96]
        inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.a1 = nn.Sequential(nn.Conv2d(256, 128, 1, 1))
        self.a2 = nn.Sequential(nn.Conv2d(128, 96, 1, 1))

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])

            if ru == 0:
                num_blocks_level = 12
            else:
                num_blocks_level = 3

            for j in range(num_blocks_level):
                # if curr_inp_resu[j]==3:
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
                # else:
                # layers.append(MyBlock(curr_inp_resu[j], nunits))

            self.layers_set_up[ru].append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))

            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            if ru == 0:
                temp = self.layers_set_final[ru](x)
                x = x + temp
            elif ru == 1:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a1(x)
                x = temp + temp2
            elif ru == 2:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a2(x)
                x = temp + temp2
            x = self.layers_set_final_up[ru](x)

        x = self.main(x)

        return x
