"""VGG11 as implemented in paper"""
from torch import nn
from torch.nn import functional as F


class VGG11(nn.Module):
    """VGG-11"""

    def __init__(self, in_channels=3, keypoints=11, fc_units=4096):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.keypoints = keypoints
        self.fc_units = fc_units
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # infeatures = input_width / (2 ** 5), since 5 max pool layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512 * 16 * 16, out_features=self.fc_units),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=self.fc_units, out_features=self.fc_units),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=self.fc_units, out_features=self.keypoints * 2),
        )

    def forward(self, x, targets=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(-1, self.keypoints, 2)

        # If given targets, calculate loss
        loss = None
        if targets is not None:
            loss = F.mse_loss(x, targets)

        return x, loss
