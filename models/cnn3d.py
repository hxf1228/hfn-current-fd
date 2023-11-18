import torch.nn as nn
import torch
from utils.smu import SMU


class Cnn3dPlain(nn.Module):

    def __init__(self, num_classes=4):
        super(Cnn3dPlain, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, padding=(1, 1, 1), bias=False)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=(1, 1, 1), bias=False)
        self.conv4 = nn.Conv3d(128, 8, 3, padding=(1, 1, 1), bias=False)

        self.pool1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool4 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))
        # self.pool3 = nn.MaxPool3d(2, stride=2)

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(8)

        # self.lrelu = nn.LeakyReLU(inplace=True)
        self.lrelu = SMU()

        # self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout3d(0.2)
        # self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)
        # Conv x1
        out = self.lrelu(self.bn1(self.conv1(out)))
        out = self.pool1(out)

        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.pool2(out)

        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.pool3(out)

        out = self.lrelu(self.bn4(self.conv4(out)))
        out = self.pool4(out)
        out = self.dropout(out)
        out = out.view(-1, 8 * 8 * 8)

        # out = self.fc1(out)
        out = self.fc2(out)

        return out


class Cnn3dFuse(nn.Module):

    def __init__(self, num_classes=4):
        super(Cnn3dFuse, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, 3, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, 3, padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))

        self.pool1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool4 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(8)

        self.lv1bn = nn.BatchNorm3d(8)
        self.lv1conv = nn.Conv3d(32, 8, 3, padding=(1, 1, 1))
        self.lv1pool = nn.AdaptiveAvgPool3d((8, 8, 2))
        self.lv2bn = nn.BatchNorm3d(8)
        self.lv2conv = nn.Conv3d(64, 8, 3, padding=(1, 1, 1))
        self.lv2pool = nn.AdaptiveAvgPool3d((8, 8, 2))
        self.lv3bn = nn.BatchNorm3d(8)
        self.lv3conv = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))
        self.lv3pool = nn.AdaptiveAvgPool3d((8, 8, 2))

        self.lrelu = nn.LeakyReLU(inplace=True)

        self.convf = nn.Conv3d(8, 4, 3, padding=(1, 1, 1))
        self.bnf = nn.BatchNorm3d(4)
        self.poolf = nn.MaxPool3d(2, stride=2)
        # self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)

        out = self.lrelu(self.bn1(self.conv1(out)))
        out = self.pool1(out)

        feature1 = self.lv1pool(self.lrelu(self.lv1bn(self.lv1conv(out))))
        # feature1 = feature1.view(-1, 4 * 8 * 8)

        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        feature2 = self.lv2pool(self.lrelu(self.lv2bn(self.lv2conv(out))))
        # feature2 = feature2.view(-1, 3 * 8 * 8)

        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        feature3 = self.lv3pool(self.lrelu(self.lv3bn(self.lv3conv(out))))
        # feature3 = feature3.view(-1, 2 * 8 * 8)

        out = self.lrelu(self.bn4(self.conv4(out)))
        # out = self.pool4(out)

        # out = out.view(-1, 64 * 4 * 4)
        multifeature = torch.cat((feature1, feature2, feature3, out), dim=4)

        final = self.lrelu(self.bnf(self.convf(multifeature)))
        final = self.poolf(final)
        final = self.dropout(final)
        final = final.view(-1, 4 * 4 * 4 * 4)
        # final = self.fc1(final)
        final = self.fc2(final)

        return final


class Cnn3dFuseSmuV1(nn.Module):

    def __init__(self, num_classes=4):
        super(Cnn3dFuseSmuV1, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, 3, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, 3, padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))

        self.pool1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool4 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(8)

        self.lv1bn = nn.BatchNorm3d(8)
        self.lv1conv = nn.Conv3d(32, 8, 3, padding=(1, 1, 1))
        self.lv1pool = nn.AdaptiveAvgPool3d((8, 8, 1))
        self.lv2bn = nn.BatchNorm3d(8)
        self.lv2conv = nn.Conv3d(64, 8, 3, padding=(1, 1, 1))
        self.lv2pool = nn.AdaptiveAvgPool3d((8, 8, 1))
        self.lv3bn = nn.BatchNorm3d(8)
        self.lv3conv = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))
        self.lv3pool = nn.AdaptiveAvgPool3d((8, 8, 1))

        self.lrelu = SMU()

        self.convf = nn.Conv3d(32, 8, 3, padding=(1, 1, 1))
        self.bnf = nn.BatchNorm3d(8)
        self.poolf = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        # self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)

        out = self.lrelu(self.bn1(self.conv1(out)))
        out = self.pool1(out)

        feature1 = self.lv1pool(self.lrelu(self.lv1bn(self.lv1conv(out))))
        # feature1 = feature1.view(-1, 4 * 8 * 8)

        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        feature2 = self.lv2pool(self.lrelu(self.lv2bn(self.lv2conv(out))))
        # feature2 = feature2.view(-1, 3 * 8 * 8)

        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        feature3 = self.lv3pool(self.lrelu(self.lv3bn(self.lv3conv(out))))
        # feature3 = feature3.view(-1, 2 * 8 * 8)

        out = self.lrelu(self.bn4(self.conv4(out)))
        out = self.pool4(out)

        # out = out.view(-1, 64 * 4 * 4)
        multifeature = torch.cat((feature1, feature2, feature3, out), dim=1)

        final = self.lrelu(self.bnf(self.convf(multifeature)))
        final = self.poolf(final)
        final = self.dropout(final)
        final = final.view(-1, 8 * 4 * 4)
        # final = self.fc1(final)
        final = self.fc2(final)

        return final


class Cnn3dFuseSmu(nn.Module):

    def __init__(self, num_classes=4):
        super(Cnn3dFuseSmu, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, 3, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, 3, padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))

        self.pool1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        # self.pool4 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(8)

        self.lv1bn = nn.BatchNorm3d(8)
        self.lv1conv = nn.Conv3d(32, 8, 3, padding=(1, 1, 1))
        self.lv1pool = nn.AdaptiveAvgPool3d((8, 8, 2))
        self.lv2bn = nn.BatchNorm3d(8)
        self.lv2conv = nn.Conv3d(64, 8, 3, padding=(1, 1, 1))
        self.lv2pool = nn.AdaptiveAvgPool3d((8, 8, 2))
        self.lv3bn = nn.BatchNorm3d(8)
        self.lv3conv = nn.Conv3d(128, 8, 3, padding=(1, 1, 1))
        self.lv3pool = nn.AdaptiveAvgPool3d((8, 8, 2))

        self.lrelu = SMU()

        self.convf = nn.Conv3d(8, 4, 3, padding=(1, 1, 1))
        self.bnf = nn.BatchNorm3d(4)
        self.poolf = nn.MaxPool3d(2, stride=2)
        # self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)

        out = self.lrelu(self.bn1(self.conv1(out)))
        out = self.pool1(out)

        feature1 = self.lv1pool(self.lrelu(self.lv1bn(self.lv1conv(out))))
        # feature1 = feature1.view(-1, 4 * 8 * 8)

        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        feature2 = self.lv2pool(self.lrelu(self.lv2bn(self.lv2conv(out))))
        # feature2 = feature2.view(-1, 3 * 8 * 8)

        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        feature3 = self.lv3pool(self.lrelu(self.lv3bn(self.lv3conv(out))))
        # feature3 = feature3.view(-1, 2 * 8 * 8)

        out = self.lrelu(self.bn4(self.conv4(out)))
        # out = self.pool4(out)

        # out = out.view(-1, 64 * 4 * 4)
        multifeature = torch.cat((feature1, feature2, feature3, out), dim=4)

        final = self.lrelu(self.bnf(self.convf(multifeature)))
        final = self.poolf(final)
        final = self.dropout(final)
        final = final.view(-1, 4 * 4 * 4 * 4)
        # final = self.fc1(final)
        final = self.fc2(final)

        return final


if __name__ == '__main__':
    input = torch.randn(32, 3, 64, 64, 2)
    model = Cnn3dFuseSmu(num_classes=4)
    prediction = model(input)
    print(prediction.shape)

#
# import torch
# import numpy as np
#
# if __name__ == '__main__':
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     #device = torch.device('cpu')
#     dummy_input = torch.randn(1, 3, 64, 64, 2, dtype=torch.float).to(device)
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     repetitions = 500
#     timings = np.zeros((repetitions, 1))
#     model = Cnn3dFuseSmu(4).to(device)
#     # GPU-WARM-UP
#     for _ in range(100):
#         _ = model(dummy_input)
#     # MEASURE PERFORMANCE
#     with torch.no_grad():
#         for rep in range(repetitions):
#             starter.record()
#             _ = model(dummy_input)
#             ender.record()
#             # WAIT FOR GPU SYNC
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)
#             timings[rep] = curr_time
#     mean_syn = np.sum(timings) / repetitions
#     std_syn = np.std(timings)
#     print(' * Mean@1 {mean_syn:.4f}ms Std@5 {std_syn:.4f}ms'.format(mean_syn=mean_syn, std_syn=std_syn))
