import torch
import torch.nn as nn
import numpy as np


class MatrixCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(MatrixCNN, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(1, 64, (2, 15), 1, padding=(0, 7))
        self.dropout1 = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d((1, 8), (1, 8))
        self.conv2 = nn.Conv2d(64, 128, (1, 8), 1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.pool2 = nn.MaxPool2d((1, 4), (1, 4))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16128, 100)  # 200 100
        self.fc2 = nn.Linear(100, 200)
        self.classifier = nn.Linear(200, num_classes)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        # out = self.dropout1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        # out = self.dropout2(out)
        out = self.pool2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)

        out = self.classifier(out)
        out = self.activation(out)

        return out


# if __name__ == '__main__':
#     input = torch.randn(32, 1, 2, 4096)
#     model = MatrixCNN(num_classes=4)
#     prediction = model(input)
#     print(prediction.shape)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 1, 2, 4096, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings = np.zeros((repetitions, 1))
    model = MatrixCNN(4).to(device)
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
