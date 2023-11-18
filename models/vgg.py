import torch.nn as nn
import torch


class VGG16(nn.Module):

    def __init__(self, num_classes=4):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv3d(3, 32, 3, padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(32, 32, 3, padding=(1, 1, 1))

        self.conv2_1 = nn.Conv3d(32, 64, 3, padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(64, 64, 3, padding=(1, 1, 1))

        self.conv3_1 = nn.Conv3d(64, 128, 3, padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, 3, padding=(1, 1, 1))
        self.conv3_3 = nn.Conv3d(128, 128, 3, padding=(1, 1, 1))

        self.conv4_1 = nn.Conv3d(128, 256, 3, padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(256, 256, 3, padding=(1, 1, 1))
        self.conv4_3 = nn.Conv3d(256, 256, 3, padding=(1, 1, 1))

        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool4 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1_1 = nn.BatchNorm3d(32)
        self.bn1_2 = nn.BatchNorm3d(32)
        self.bn2_1 = nn.BatchNorm3d(64)
        self.bn2_2 = nn.BatchNorm3d(64)
        self.bn3_1 = nn.BatchNorm3d(128)
        self.bn3_2 = nn.BatchNorm3d(128)
        self.bn3_3 = nn.BatchNorm3d(128)
        self.bn4_1 = nn.BatchNorm3d(256)
        self.bn4_2 = nn.BatchNorm3d(256)
        self.bn4_3 = nn.BatchNorm3d(256)

        self.act_func = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)

        # Conv x1
        out = self.act_func(self.bn1_1(self.conv1_1(out)))
        out = self.act_func(self.bn1_2(self.conv1_2(out)))
        out = self.pool1(out)

        out = self.act_func(self.bn2_1(self.conv2_1(out)))
        out = self.act_func(self.bn2_2(self.conv2_2(out)))
        out = self.pool2(out)

        out = self.act_func(self.bn3_1(self.conv3_1(out)))
        out = self.act_func(self.bn3_2(self.conv3_2(out)))
        out = self.act_func(self.bn3_2(self.conv3_3(out)))
        out = self.pool3(out)

        out = self.act_func(self.bn4_1(self.conv4_1(out)))
        out = self.act_func(self.bn4_2(self.conv4_2(out)))
        out = self.act_func(self.bn4_2(self.conv4_3(out)))
        out = self.pool4(out)

        out = self.dropout(out)
        out = out.view(-1, 256 * 4 * 4)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


if __name__ == '__main__':
    input = torch.randn(32, 3, 64, 64, 2)
    model = VGG16(num_classes=4)
    prediction = model(input)
    print(prediction.shape)
