"""
Method for registering and building any models in ./models/*
"""
from dl_schema.models.resnet18 import ResNet18
from dl_schema.models.vgg11 import VGG11
from dl_schema.models.babycnn import BabyCNN


MODEL_REGISTRY = {"ResNet18": ResNet18, "VGG11": VGG11}


def build_model(model_class, cfg):
    """initialize model from class and model cfg. class must be in global scope"""
    class_ = globals()[model_class]
    return class_(cfg)


if __name__ == "__main__":
    """Test the model"""
    from torch.utils.data import DataLoader
    from torchinfo import summary
    from dl_schema.dataset import MNISTDataset
    from dl_schema.cfg import TrainConfig

    cfg = TrainConfig()
    train_data = MNISTDataset(split="train", cfg=cfg)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    x, y = next(iter(train_dataloader))

    model = BabyCNN(cfg.model)
    y_pred = model(x)
    summary(model, (64, 1, 28, 28))
    print(f"Input shape: {x.size()}")
    print(f"Labels shape: {y.size()}")
    print(f"Inference shape: {y_pred.size()}")
    print(f"y_true: {y}")
    print(f"y_pred: {y_pred}")
