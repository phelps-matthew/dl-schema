"""
Sample convolutional model.
"""
import torch
from torchvision import models
import torch.nn as nn
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

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being
        very defensive: We are separating out all parameters of the model into two
        buckets: those that will experience weight decay for regularization and those
        that won't (biases, and layernorm/embedding weights).  We are then returning the
        PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.BatchNorm2d,)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.lr, betas=train_config.betas
        )
        return optimizer

    def forward(self, x, targets=None):
        x = self.feature_extractor(x)
        # Flatten extra dimensions after average pooling layer.
        x = x.view(x.size(0), -1)

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

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being
        very defensive: We are separating out all parameters of the model into two
        buckets: those that will experience weight decay for regularization and those
        that won't (biases, and layernorm/embedding weights).  We are then returning the
        PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.BatchNorm2d,)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.lr, betas=train_config.betas
        )
        return optimizer

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


MODEL_REGISTRY = {"ResNet18": ResNet18, "VGG11": VGG11}


def build_model(cfg_dict):
    name = cfg_dict.pop("name")
    return MODEL_REGISTRY[name](**cfg_dict)


if __name__ == "__main__":
    """Test the model"""
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from dl_schema.dataset import MyDataset
    from torchinfo import summary

    # Form dataloader. ToTensor scales image pixels to [0.0, 1.0] floats.
    train_data = MyDataset(split="train", transform=ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    x, y = next(iter(train_dataloader))

    model = ResNet18()
    y_pred, _ = model(x)
    summary(model, (64, 3, 512, 512))
    print(f"Input shape: {x.size()}")
    print(f"Labels shape: {y.size()}")
    print(f"Inference shape: {y_pred.size()}")
    print(f"y_true: {y}")
    print(f"y_pred: {y_pred}")
