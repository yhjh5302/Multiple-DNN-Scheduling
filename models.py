import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import math

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, transform_input: bool = True, init_weights: bool = True) -> None:
        super(GoogLeNet, self).__init__()
        self.transform_input = transform_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.conv1_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.conv3_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a_branch1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch2 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch3 = nn.Sequential(
            nn.Conv2d(192, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception3b_maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a_branch1 = nn.Sequential(
            nn.Conv2d(480, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch2 = nn.Sequential(
            nn.Conv2d(480, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 208, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(208, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch3 = nn.Sequential(
            nn.Conv2d(480, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(480, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch1 = nn.Sequential(
            nn.Conv2d(512, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch2 = nn.Sequential(
            nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(112, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(112, 224, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(224, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch3 = nn.Sequential(
            nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch3 = nn.Sequential(
            nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(24, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4c_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch1 = nn.Sequential(
            nn.Conv2d(512, 112, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(112, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch2 = nn.Sequential(
            nn.Conv2d(512, 144, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(144, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 288, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(288, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch3 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4d_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch1 = nn.Sequential(
            nn.Conv2d(528, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch2 = nn.Sequential(
            nn.Conv2d(528, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch3 = nn.Sequential(
            nn.Conv2d(528, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(528, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception4e_maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception5a_branch1 = nn.Sequential(
            nn.Conv2d(832, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch2 = nn.Sequential(
            nn.Conv2d(832, 160, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(160, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch3 = nn.Sequential(
            nn.Conv2d(832, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5a_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch1 = nn.Sequential(
            nn.Conv2d(832, 384, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(384, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch2 = nn.Sequential(
            nn.Conv2d(832, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch3 = nn.Sequential(
            nn.Conv2d(832, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        self.inception5b_branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(832, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
        )
        '''
        self.inception_aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
        )
        self.inception_aux2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(528, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
        )
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)
        # conv1 and max pool()
        x = self.conv1(x)
        x = self.conv1_maxpool(x)
        # conv2()
        x = self.conv2(x)
        # conv3 and max pool()
        x = self.conv3(x)
        x = self.conv3_maxpool(x)
        # inception3a()
        branch1 = self.inception3a_branch1(x)
        branch2 = self.inception3a_branch2(x)
        branch3 = self.inception3a_branch3(x)
        branch4 = self.inception3a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception3b and max pool()
        branch1 = self.inception3b_branch1(x)
        branch2 = self.inception3b_branch2(x)
        branch3 = self.inception3b_branch3(x)
        branch4 = self.inception3b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception3b_maxpool(x)
        # inception4a()
        branch1 = self.inception4a_branch1(x)
        branch2 = self.inception4a_branch2(x)
        branch3 = self.inception4a_branch3(x)
        branch4 = self.inception4a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4b()
        branch1 = self.inception4b_branch1(x)
        branch2 = self.inception4b_branch2(x)
        branch3 = self.inception4b_branch3(x)
        branch4 = self.inception4b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4c()
        branch1 = self.inception4c_branch1(x)
        branch2 = self.inception4c_branch2(x)
        branch3 = self.inception4c_branch3(x)
        branch4 = self.inception4c_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4d()
        branch1 = self.inception4d_branch1(x)
        branch2 = self.inception4d_branch2(x)
        branch3 = self.inception4d_branch3(x)
        branch4 = self.inception4d_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception4e()
        branch1 = self.inception4e_branch1(x)
        branch2 = self.inception4e_branch2(x)
        branch3 = self.inception4e_branch3(x)
        branch4 = self.inception4e_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = self.inception4e_maxpool(x)
        # inception5a()
        branch1 = self.inception5a_branch1(x)
        branch2 = self.inception5a_branch2(x)
        branch3 = self.inception5a_branch3(x)
        branch4 = self.inception5a_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # inception5b()
        branch1 = self.inception5b_branch1(x)
        branch2 = self.inception5b_branch2(x)
        branch3 = self.inception5b_branch3(x)
        branch4 = self.inception5b_branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        # avg pool, flatten and fully_connected
        x = self.avgpool(x)
        x = self.fully_connected(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.separable_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.separable_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.separable_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.separable_conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.separable_conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.separable_conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.separable_conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.separable_conv14 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, 1000)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        x = self.separable_conv6(x)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        x = self.separable_conv11(x)
        x = self.separable_conv12(x)
        x = self.separable_conv13(x)
        x = self.separable_conv14(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class VGGNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(VGGNet, self).__init__()
        self.features1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features1_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features2_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features2_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features3_conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features4_conv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features5_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Dropout(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1_conv1(x)
        x = self.features1_conv2(x)
        x = self.features2_conv1(x)
        x = self.features2_conv2(x)
        x = self.features3_conv1(x)
        x = self.features3_conv2(x)
        x = self.features3_conv3(x)
        x = self.features4_conv1(x)
        x = self.features4_conv2(x)
        x = self.features4_conv3(x)
        x = self.features5_conv1(x)
        x = self.features5_conv2(x)
        x = self.features5_conv3(x)
        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet101(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(ResNet101, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1_bottleneck0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_bottleneck1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_bottleneck2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer2_bottleneck0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer3_bottleneck0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck5 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck6 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck7 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck9 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck10 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck11 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck12 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck13 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck14 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck15 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck16 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck17 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck18 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck19 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck20 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck21 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck22 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer4_bottleneck0 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_bottleneck1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_bottleneck2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)

        identity = x
        x = self.layer1_bottleneck0(x)
        x += self.layer1_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer1_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer1_bottleneck2(x)
        x += identity
        x = self.relu(x)
        
        identity = x
        x = self.layer2_bottleneck0(x)
        x += self.layer2_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck2(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck3(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.layer3_bottleneck0(x)
        x += self.layer3_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck2(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck3(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck4(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck5(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck6(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck7(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck8(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck9(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck10(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck11(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck12(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck13(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck14(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck15(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck16(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck17(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck18(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck19(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck20(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck21(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck22(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.layer4_bottleneck0(x)
        x += self.layer4_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer4_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer4_bottleneck2(x)
        x += identity
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class ResNet152(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True ) -> None:
        super(ResNet152, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1_bottleneck0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_bottleneck1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_bottleneck2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer2_bottleneck0 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2_bottleneck7 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer3_bottleneck0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck5 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck6 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck7 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck9 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck10 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck11 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck12 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck13 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck14 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck15 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck16 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck17 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck18 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck19 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck20 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck21 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck22 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck23 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck24 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck25 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck26 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck27 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck28 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck29 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck30 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck31 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck32 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck33 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck34 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer3_bottleneck35 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer4_bottleneck0 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_bottleneck1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer4_bottleneck2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)

        identity = x
        x = self.layer1_bottleneck0(x)
        x += self.layer1_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer1_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer1_bottleneck2(x)
        x += identity
        x = self.relu(x)
        
        identity = x
        x = self.layer2_bottleneck0(x)
        x += self.layer2_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck2(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck3(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck4(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck5(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck6(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer2_bottleneck7(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.layer3_bottleneck0(x)
        x += self.layer3_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck2(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck3(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck4(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck5(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck6(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck7(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck8(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck9(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck10(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck11(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck12(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck13(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck14(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck15(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck16(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck17(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck18(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck19(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck20(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck21(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck22(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck23(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck24(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck25(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck26(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck27(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck28(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck29(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck30(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck31(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck32(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck33(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck34(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer3_bottleneck35(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.layer4_bottleneck0(x)
        x += self.layer4_downsample(identity)
        x = self.relu(x)
        identity = x
        x = self.layer4_bottleneck1(x)
        x += identity
        x = self.relu(x)
        identity = x
        x = self.layer4_bottleneck2(x)
        x += identity
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)