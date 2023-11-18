import torch.nn as nn
import torch
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(15)
        self.bn4 = nn.BatchNorm2d(30)
        self.conv1 = nn.Conv2d(1, 5, 9, 1, padding=4)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(5, 10, 7, 1, padding=3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(10, 15, 5, 1, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(15, 30, 3, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        # self.dropout = nn.Dropout2d(0.1)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(480, num_classes)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        # out = self.dropout(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)
        out = self.pool4(out)

        out = self.flatten(out)
        out = self.fc(out)
        # out = self.activation(out)

        return out


class GrayCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(GrayCNN, self).__init__()
        self.conv1b = CNN(num_classes)
        self.conv2b = CNN(num_classes)
        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(5, num_classes)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        x1 = x[:, 0, :, :].view(batch, 1, 64, 64)
        x2 = x[:, 1, :, :].view(batch, 1, 64, 64)
        out1 = self.conv1b(x1)
        out2 = self.conv2b(x2)

        out = torch.cat([out1, out2], dim=1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


# if __name__ == '__main__':
#     input = torch.randn(32, 2, 64, 64)
#     model = GrayCNN(num_classes=4)
#     prediction = model(input)
#     print(prediction.shape)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dummy_input = torch.randn(1, 2, 64, 64, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings = np.zeros((repetitions, 1))
    model = GrayCNN(4).to(device)
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
