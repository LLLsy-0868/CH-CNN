from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import utils


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, num_group=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=1 if in_planes==3 else num_group)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Basic(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, num_group=1):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=num_group)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, in_planes, planes, stride=1,num_group=1):
        super(Bottleneck, self).__init__()
        width = int(planes/4)
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False,groups=1 if in_planes==3 else num_group)
        self.bn1 = nn.BatchNorm2d(width)
        if num_group == 4:
            num_group = width
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt0(nn.Module):

    def __init__(self, in_planes, planes, num_group=4, stride=1):
        super(ResNeXt0, self).__init__()
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, num_group=num_group))
        self.layer =nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt1(nn.Module):

    def __init__(self, in_planes, planes, num_group=4, stride=1):
        super(ResNeXt1, self).__init__()
        layers = []
        layers.append(Bottleneck(in_planes, planes, stride, num_group=num_group))
        self.layer =nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self,indi):
        super(EvoCNNModel, self).__init__()
        units = indi.units
        for i in range(len(units)):
            if units[i].type==1:
                if units[i].block_type==0:
                    conv=Basic(units[i].in_channel,units[i].out_channel)

                elif units[i].block_type==1:
                    conv=BasicBlock(units[i].in_channel,units[i].out_channel)

                elif units[i].block_type==2:
                    conv=Bottleneck(units[i].in_channel,units[i].out_channel)

                elif units[i].block_type==3:
                    conv=ResNeXt0(units[i].in_channel,units[i].out_channel)

                elif units[i].block_type==4:
                    conv = ResNeXt1(units[i].in_channel,units[i].out_channel)
                self.add_module("conv{}_{}_{}".format(i,units[i].in_channel, units[i].out_channel), conv)

            else:
                # print(i, units[i].type)
                if units[i].max_or_avg< 0.5:
                    pool=nn.MaxPool2d(2)

                else:
                    pool=nn.AvgPool2d(2)
                self.add_module("pool{}".format(i), pool)


        out_channel_list = []
        image_output_size = 224
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            else:
                out_channel_list.append(out_channel_list[-1])
                image_output_size = int(image_output_size / 2)
        full=nn.Linear(image_output_size * image_output_size * out_channel_list[-1],utils.StatusUpdateTool.get_num_class())
        self.add_module("full",full)


        #generated_init


    def forward(self, x):
        #generate_forward

        for name, module in self.named_children():
            # print(name)
            if 'full' in name :
                x=x.view(x.size(0),-1)
            if 'pool' in name:
                x = F.max_pool2d(x, 2)
            else:
                x= module(x)
        return x


