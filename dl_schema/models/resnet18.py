"""torchvision ResNet18"""
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNet18(nn.Module):
    """Sample ResNet model."""

    def __init__(self, in_channels=3, keypoints=11, fc_units=4096, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.keypoints = keypoints
        self.fc_units = fc_units
        resnet18 = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*nn.ModuleList(resnet18.children())[:-1])
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=resnet18.fc.in_features, out_features=self.fc_units),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=self.fc_units, out_features=self.fc_units),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=self.fc_units, out_features=self.keypoints * 2),
        )

    def forward(self, x, targets=None):
        x = self.feature_extractor(x)
        # Flatten extra dimensions after average pooling layer.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(-1, self.keypoints, 2)

        # If given targets, calculate loss
        loss = None
        if targets is not None:
            loss = F.mse_loss(x, targets)

        return x, loss
