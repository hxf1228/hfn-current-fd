import torch
import torch.nn as nn


def Conv1(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
                         nn.BatchNorm3d(out_channel),
                         nn.ReLU())


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = Conv1(in_channel=3, out_channel=32, kernel_size=3, stride=2, padding=(1, 1, 1))
        self.conv2 = Conv1(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.conv3 = Conv1(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.branch1_1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.branch1_2 = Conv1(in_channel=64, out_channel=96, kernel_size=3, stride=(2, 2, 1),
                               padding=(1, 1, 1))
        self.branch2_1_1 = Conv1(in_channel=160, out_channel=64, kernel_size=1, stride=1,
                                 padding=(1, 1, 1))
        self.branch2_1_2 = Conv1(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)
        self.branch2_2_1 = Conv1(in_channel=160, out_channel=64, kernel_size=1, stride=1,
                                 padding=(1, 1, 1))
        self.branch2_2_2 = Conv1(in_channel=64, out_channel=64, kernel_size=(7, 1, 1), stride=1,
                                 padding=(3, 0, 0))
        self.branch2_2_3 = Conv1(in_channel=64, out_channel=64, kernel_size=(1, 1, 7), stride=1,
                                 padding=(0, 0, 3))
        self.branch2_2_4 = Conv1(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=0)
        self.branch3_1 = Conv1(in_channel=192, out_channel=192, kernel_size=3, stride=2,
                               padding=(1, 1, 1))
        self.branch3_2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4_1 = self.branch1_1(out3)
        out4_2 = self.branch1_2(out3)
        out4 = torch.cat((out4_1, out4_2), dim=1)
        out5_1 = self.branch2_1_2(self.branch2_1_1(out4))
        out5_2 = self.branch2_2_4(self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4))))
        out5 = torch.cat((out5_1, out5_2), dim=1)
        out6_1 = self.branch3_1(out5)
        out6_2 = self.branch3_2(out5)
        out = torch.cat((out6_1, out6_2), dim=1)
        return out


class InceptionA(nn.Module):
    def __init__(self, in_channel):
        super(InceptionA, self).__init__()
        # self.branch1_1 = nn.AvgPool3d(kernel_size=2, stride=1, padding=(1, 1, 1))
        self.branch1_2 = Conv1(in_channel=in_channel, out_channel=96, kernel_size=1, stride=1,
                               padding=(0, 0, 0))
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=96, kernel_size=1, stride=1,
                               padding=(0, 0, 0))
        self.branch3_1 = Conv1(in_channel=in_channel, out_channel=64, kernel_size=1, stride=1,
                               padding=(0, 0, 0))
        self.branch3_2 = Conv1(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.branch4_1 = Conv1(in_channel=in_channel, out_channel=64, kernel_size=1, stride=1,
                               padding=(0, 0, 0))
        self.branch4_2 = Conv1(in_channel=64, out_channel=96, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.branch4_3 = Conv1(in_channel=96, out_channel=96, kernel_size=3, stride=1, padding=(1, 1, 1))

    def forward(self, x):
        out1 = self.branch1_2(x)
        out2 = self.branch2_1(x)
        out3 = self.branch3_2(self.branch3_1(x))
        out4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))
        return torch.cat((out1, out2, out3, out4), dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channel):
        super(InceptionB, self).__init__()
        # self.branch1_1 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.branch1_2 = Conv1(in_channel=in_channel, out_channel=128, kernel_size=1, stride=1,
                               padding=0)
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=384, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_2 = Conv1(in_channel=192, out_channel=224, kernel_size=(1, 1, 7), stride=1,
                               padding=(0, 0, 3))
        self.branch3_3 = Conv1(in_channel=224, out_channel=256, kernel_size=(7, 1, 1), stride=1,
                               padding=(3, 0, 0))
        self.branch4_1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1,
                               padding=0)
        self.branch4_2 = Conv1(in_channel=192, out_channel=192, kernel_size=(1, 1, 7), stride=1,
                               padding=(0, 0, 3))
        self.branch4_3 = Conv1(in_channel=192, out_channel=224, kernel_size=(7, 1, 1), stride=1,
                               padding=(3, 0, 0))
        self.branch4_4 = Conv1(in_channel=224, out_channel=224, kernel_size=(1, 1, 7), stride=1,
                               padding=(0, 0, 3))
        self.branch4_5 = Conv1(in_channel=224, out_channel=256, kernel_size=(7, 1, 1), stride=1,
                               padding=(3, 0, 0))

    def forward(self, x):
        out1 = self.branch1_2(x)
        out2 = self.branch2_1(x)
        out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        out4 = self.branch4_5(self.branch4_4(self.branch4_3(self.branch4_2(self.branch4_1(x)))))
        return torch.cat((out1, out2, out3, out4), dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channel):
        super(InceptionC, self).__init__()
        # self.branch1_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch1_2 = Conv1(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1,
                               padding=0)
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_1 = Conv1(in_channel=in_channel, out_channel=384, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_2_1 = Conv1(in_channel=384, out_channel=256, kernel_size=(1, 1, 3), stride=1,
                                 padding=(0, 0, 1))
        self.branch3_2_2 = Conv1(in_channel=384, out_channel=256, kernel_size=(3, 1, 1), stride=1,
                                 padding=(1, 0, 0))
        self.branch4_1 = Conv1(in_channel=in_channel, out_channel=384, kernel_size=1, stride=1,
                               padding=0)
        self.branch4_2 = Conv1(in_channel=384, out_channel=448, kernel_size=(1, 1, 3), stride=1,
                               padding=(0, 0, 1))
        self.branch4_3 = Conv1(in_channel=448, out_channel=512, kernel_size=(3, 1, 1), stride=1,
                               padding=(1, 0, 0))
        self.branch4_4_1 = Conv1(in_channel=512, out_channel=256, kernel_size=(3, 1, 1), stride=1,
                                 padding=(1, 0, 0))
        self.branch4_4_2 = Conv1(in_channel=512, out_channel=256, kernel_size=(1, 1, 3), stride=1,
                                 padding=(0, 0, 1))

    def forward(self, x):
        out1 = self.branch1_2(x)
        out2 = self.branch2_1(x)
        out3_1 = self.branch3_1(x)
        out3_2_1 = self.branch3_2_1(out3_1)
        out3_2_2 = self.branch3_2_2(out3_1)
        out3 = torch.cat((out3_2_1, out3_2_2), dim=1)
        out4_1 = self.branch4_3(self.branch4_2(self.branch4_1(x)))
        out4_2_1 = self.branch4_4_1(out4_1)
        out4_2_2 = self.branch4_4_2(out4_1)
        out4 = torch.cat((out4_2_1, out4_2_2), dim=1)
        return torch.cat((out1, out2, out3, out4), dim=1)


class ReductionA(nn.Module):
    def __init__(self, in_channel):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0))
        self.branch2 = Conv1(in_channel=in_channel, out_channel=384, kernel_size=3, stride=2,
                             padding=(1, 1, 1))
        self.branch3_1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_2 = Conv1(in_channel=192, out_channel=224, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = Conv1(in_channel=224, out_channel=256, kernel_size=3, stride=2,
                               padding=(1, 1, 1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        return torch.cat((out1, out2, out3), dim=1)


class ReductionB(nn.Module):
    def __init__(self, in_channel):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(0, 0, 1))
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=192, kernel_size=1, stride=1,
                               padding=(1, 1, 1))
        self.branch2_2 = Conv1(in_channel=192, out_channel=192, kernel_size=3, stride=2, padding=0)
        self.branch3_1 = Conv1(in_channel=in_channel, out_channel=256, kernel_size=1, stride=1,
                               padding=0)
        self.branch3_2 = Conv1(in_channel=256, out_channel=256, kernel_size=(1, 1, 7), stride=1,
                               padding=(0, 0, 3))
        self.branch3_3 = Conv1(in_channel=256, out_channel=320, kernel_size=(7, 1, 1), stride=1,
                               padding=(3, 0, 0))
        self.branch3_4 = Conv1(in_channel=320, out_channel=320, kernel_size=3, stride=2,
                               padding=(1, 1, 1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2_2(self.branch2_1(x))
        out3 = self.branch3_4(self.branch3_3(self.branch3_2(self.branch3_1(x))))
        return torch.cat((out1, out2, out3), dim=1)


class InceptionV4(nn.Module):

    def __init__(self, num_classes=4):
        super(InceptionV4, self).__init__()
        self.Stem = Stem()
        self.InceptionA1 = InceptionA(in_channel=384)
        self.InceptionA2 = InceptionA(in_channel=384)
        self.InceptionA3 = InceptionA(in_channel=384)
        self.InceptionA4 = InceptionA(in_channel=384)
        self.ReductionA = ReductionA(in_channel=384)
        self.InceptionB1 = InceptionB(in_channel=1024)
        self.InceptionB2 = InceptionB(in_channel=1024)
        self.InceptionB3 = InceptionB(in_channel=1024)
        self.InceptionB4 = InceptionB(in_channel=1024)
        self.InceptionB5 = InceptionB(in_channel=1024)
        self.InceptionB6 = InceptionB(in_channel=1024)
        self.InceptionB7 = InceptionB(in_channel=1024)
        self.ReductionB = ReductionB(in_channel=1024)
        self.InceptionC1 = InceptionC(in_channel=1536)
        self.InceptionC2 = InceptionC(in_channel=1536)
        self.InceptionC3 = InceptionC(in_channel=1536)
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1536, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Stem(x)
        out = self.InceptionA1(out)
        # out = self.InceptionA2(out)
        # out = self.InceptionA3(out)
        out = self.ReductionA(out)
        out = self.InceptionB1(out)
        # out = self.InceptionB2(out)
        # out = self.InceptionB3(out)
        # out = self.InceptionB4(out)
        # out = self.InceptionB5(out)
        # out = self.InceptionB6(out)
        # out = self.InceptionB7(out)
        out = self.ReductionB(out)
        out = self.InceptionC1(out)
        # out = self.InceptionC2(out)
        # out = self.InceptionC3(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.fc(out.view(out.size(0), -1))
        return out


if __name__ == '__main__':
    # input = torch.randn(32, 3, 299, 299)
    input = torch.randn(32, 3, 64, 64, 2)
    model = InceptionV4(num_classes=4)
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
