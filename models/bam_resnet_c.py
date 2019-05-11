# Implementation of BAM with only channel attention in ResNet50, ResNet34

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['BAM_ResNet', 'bam_resnet34_c', 'bam_resnet50_c']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BAM_Block(nn.Module):
    r = 16
    d = 4
    def __init__(self, inplanes):
        super(BAM_Block, self).__init__()
        norm_layer = nn.BatchNorm2d

        # Channel-attention: 
        self.avg_Gpool_c = nn.AdaptiveAvgPool2d(1)
        self.conv1_c = conv1x1(inplanes, inplanes/self.r)
        self.bn1_c = norm_layer(inplanes/self.r)
        self.relu1_c = nn.ReLU(inplace=True)
        self.conv2_c = conv1x1(inplanes/self.r, inplanes)

        # # Spatial-attention:
        # self.conv1_s = conv1x1(inplanes, inplanes/self.r)
        # self.bn1_s = norm_layer(inplanes/self.r)
        # self.relu1_s = nn.ReLU(inplace=True)

        # self.conv2_s = conv3x3(inplanes/self.r, inplanes/self.r, 1, 1, self.d)
        # self.bn2_s = norm_layer(inplanes/self.r)
        # self.relu2_s = nn.ReLU(inplace=True)

        # self.conv3_s = conv3x3(inplanes/self.r, inplanes/self.r, 1, 1, self.d)
        # self.bn3_s = norm_layer(inplanes/self.r)
        # self.relu3_s = nn.ReLU(inplace=True)

        # self.conv4_s = conv1x1(inplanes/self.r, 1)

        # Combine two attention branches:
        self.sigmoid_cs = nn.Sigmoid()

    def forward(self, x):
        # Channel-attention:
        out_c = self.avg_Gpool_c(x)
        out_c = self.conv1_c(out_c)
        out_c = self.bn1_c(out_c)
        out_c = self.relu1_c(out_c)
        out_c = self.conv2_c(out_c)

        # # Spatial-attention:
        # out_s = self.conv1_s(x)
        # out_s = self.bn1_s(out_s)
        # out_s = self.relu1_s(out_s)

        # out_s = self.conv2_s(out_s)
        # out_s = self.bn2_s(out_s)
        # out_s = self.relu2_s(out_s)

        # out_s = self.conv3_s(out_s)
        # out_s = self.bn3_s(out_s)
        # out_s = self.relu3_s(out_s)

        # out_s = self.conv4_s(out_s)

        # Combine two attention branches (element-wise summation)
        # out_cs = self.sigmoid_cs(out_c.expand_as(x) + out_s.expand_as(x))
        out_c = self.sigmoid_cs(out_c.expand_as(x))
        # out_s = self.bam_sigmoid_cs(out_s.expand_as(x))

        # return x * (1 + out_c)
        # return x * (1 + out_s)
        return x * (1 + out_c)

class BAM_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BAM_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BAM_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BAM_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BAM_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(BAM_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # BAM
        self.bam1 = BAM_Block(64*block.expansion)
        self.bam2 = BAM_Block(128*block.expansion)
        self.bam3 = BAM_Block(256*block.expansion)

        # make layers in ResNet arch
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.bam1(x)

        x = self.layer2(x)
        x = self.bam2(x)

        x = self.layer3(x)
        x = self.bam3(x)

        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def bam_resnet34_c(**kwargs):
    """Constructs a ResNet-34 model."""
    model = BAM_ResNet(BAM_BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def bam_resnet50_c(**kwargs):
    """Constructs a ResNet-50 model."""
    model = BAM_ResNet(BAM_Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
