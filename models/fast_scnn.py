"""Fast Segmentation Convolutional Neural Network for FireDataset"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size=3, stride=stride, padding=1,
                      groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dw_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck"""
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, kernel_size=1),
            nn.Conv2d(in_channels * t, in_channels * t, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.out = _ConvBNReLU(in_channels + 4 * inter_channels, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size, mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size, mode='bilinear', align_corners=False)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size, mode='bilinear', align_corners=False)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        return self.out(x)


class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor Module"""
    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, num_blocks=(3, 3, 3), t=6):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, stride=2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, stride=2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, stride=1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, in_channels, out_channels, blocks, t=6, stride=1):
        layers = [block(in_channels, out_channels, t, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FastSCNN(nn.Module):
    """Fast-SCNN"""
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = _DSConv(3, 64, stride=2)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128)
        self.feature_fusion = _DSConv(64, 128)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.learning_to_downsample(x)
        x = self.global_feature_extractor(x)
        x = self.feature_fusion(x)
        return F.interpolate(self.classifier(x), size, mode='bilinear', align_corners=False)


def get_fast_scnn(num_classes, **kwargs):
    return FastSCNN(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    model = get_fast_scnn(num_classes=2)
    print(model)
