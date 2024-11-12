import torch
from efficientnet_pytorch import EfficientNet


class AIModel(torch.nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", "efficientnet-b0.pth", num_classes=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x


# import torch
# import torchvision.models as models


# class AIModel(torch.nn.Module):
#     def __init__(self):
#         super(AIModel, self).__init__()
#         self.resnet = models.resnet152()
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = torch.nn.Linear(num_ftrs, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.sigmoid(x)
#         return x


# import torch
# import torch.nn as nn


# class SiLU(nn.Module):  # 28
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# class BottleneckBlock(nn.Module):
#     expansion = 4

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BottleneckBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False,
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(
#             out_channels, out_channels * self.expansion, kernel_size=1, bias=False
#         )
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.silu = SiLU()
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.silu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.silu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.silu(out)

#         return out


# class AIModel(nn.Module):
#     def __init__(self):
#         super(AIModel, self).__init__()
#         self.in_channels = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.silu = SiLU()
#         self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self.make_layer(BottleneckBlock, 64, 3)  # 增加层数
#         self.layer2 = self.make_layer(BottleneckBlock, 128, 4, stride=2)  # 增加层数
#         self.layer3 = self.make_layer(BottleneckBlock, 256, 6, stride=2)  # 增加层数
#         self.layer4 = self.make_layer(BottleneckBlock, 512, 3, stride=2)  # 增加层数

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(512 * BottleneckBlock.expansion, 256)  # 增加全连接层
#         self.fc2 = nn.Linear(256, 64)  # 增加全连接层
#         self.fc3 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()

#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_channels,
#                     out_channels * block.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.silu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.silu(x)
#         x = self.fc2(x)
#         x = self.silu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)

#         return x


# import torch.nn as nn
# import timm


# class AIModel(nn.Module):
#     def __init__(self):
#         super(AIModel, self).__init__()
#         self.swin_transformer = timm.create_model(
#             "swin_base_patch4_window12_384", num_classes=1
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.swin_transformer(x)
#         x = self.sigmoid(x)
#         return x
