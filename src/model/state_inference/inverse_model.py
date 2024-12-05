import torch.nn as nn


class CnnClassifier(nn.Module):
    def __init__(self, in_channels, n_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = nn.functional.elu(self.conv1(x))
        x = nn.functional.elu(self.conv2(x))
        x = nn.functional.elu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
