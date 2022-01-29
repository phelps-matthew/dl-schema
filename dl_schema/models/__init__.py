"""
Method for registering and building any models in /models
"""
from dl_schema.models.resnet18 import ResNet18
from dl_schema.models.vgg11 import VGG11


MODEL_REGISTRY = {"ResNet18": ResNet18, "VGG11": VGG11}


def build_model(cfg_dict):
    """initialize model from config dict"""
    name = cfg_dict.pop("name")
    return MODEL_REGISTRY[name](**cfg_dict)


if __name__ == "__main__":
    """Test the model"""
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from torchinfo import summary
    from dl_schema.dataset import MyDataset

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
