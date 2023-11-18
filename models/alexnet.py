import torch.nn as nn
import torch


class AlexNet(nn.Module):

    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 48, 11, padding=(1, 1, 5))
        self.conv2 = nn.Conv3d(48, 64, 5, padding=(1, 1, 2))
        self.conv3 = nn.Conv3d(64, 96, 3, padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(96, 32, 3, padding=(1, 1, 1))

        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))

        self.bn0 = nn.BatchNorm3d(3)
        self.bn1 = nn.BatchNorm3d(48)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(96)
        self.bn4 = nn.BatchNorm3d(32)

        self.act_func = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout3d(0.2)
        # self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        out = self.bn0(x)
        # Conv x1
        out = self.act_func(self.bn1(self.conv1(out)))
        out = self.pool1(out)

        out = self.act_func(self.bn2(self.conv2(out)))
        out = self.pool2(out)

        out = self.act_func(self.bn3(self.conv3(out)))
        # out = self.pool3(out)

        out = self.act_func(self.bn4(self.conv4(out)))
        out = self.pool3(out)
        out = self.dropout(out)
        out = out.view(-1, 32 * 6 * 6)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


if __name__ == '__main__':
    input = torch.randn(32, 3, 64, 64, 2)
    model = AlexNet(num_classes=4)
    prediction = model(input)
    print(prediction.shape)

# if __name__ == '__main__':
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # device = torch.device('cpu')
#     dummy_input = torch.randn(1, 3, 64, 64, 2, dtype=torch.float).to(device)
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     repetitions = 500
#     timings = np.zeros((repetitions, 1))
#     model = AlexNet(4).to(device)
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
