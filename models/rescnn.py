"""
Created on: 2021-10-02 22:14:48 UTC+8
@File: rescnn.py
@Description: The python version of rescnn for ITII paper
"Intelligent Mechanical Fault Diagnosis Using Multi-Sensor Fusion and Convolution Neural Network"
@Author: Xufeng Huang (huangxufeng@hust.edu.cn)
         Tingli Xie (txie67@gatech.edu)
@Copy Right: Licensed under the MIT License.
"""
import torch
import numpy as np
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.lrelu(out)
        return out


class ImprovedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedConvBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv = conv1x1(in_channels, out_channels, stride=2)

    def forward(self, x):
        residual = self.conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.lrelu(out)
        return out


class ResCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResCNN, self).__init__()
        self.in_channels = 16
        self.bn0 = nn.BatchNorm2d(3)
        self.conv = conv3x3(3, 16, stride=1)
        self.bn = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.iresblock1 = ImprovedResidualBlock(16, 16, stride=1)
        self.iresblock2 = self.iresblock1
        self.iresblock3 = ImprovedResidualBlock(32, 32, stride=1)
        self.iresblock4 = ImprovedResidualBlock(64, 64, stride=1)
        self.iconvblock1 = ImprovedConvBlock(16, 32)
        self.iconvblock2 = ImprovedConvBlock(32, 64)
        self.globalavgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * 9 * 9, num_classes)

    def forward(self, x):
        input_bn = self.bn0(x)
        # Conv x1
        out = self.conv(input_bn)
        out = self.bn(out)
        out = self.lrelu(out)
        # Improved Residual Block x 2
        out = self.iresblock1(out)
        out = self.iresblock2(out)
        out = self.iconvblock1(out)
        out = self.iresblock3(out)
        out = self.iconvblock2(out)
        out = self.iresblock4(out)
        out = self.globalavgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings = np.zeros((repetitions, 1))
    model = ResCNN(4).to(device)
    # GPU-WARM-UP
    for _ in range(100):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(' * Mean@1 {mean_syn:.4f}ms Std@5 {std_syn:.4f}ms'.format(mean_syn=mean_syn, std_syn=std_syn))
