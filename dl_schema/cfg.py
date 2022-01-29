"""Sample configurations for experiment."""
from typing import List, Callable, Optional, Literal
from dataclasses import dataclass, field
from torchvision.transforms import ToTensor
import torchvision
import pyrallis


@dataclass()
class ModelConfig:
    """config for model specification"""

    # name of model
    name: Literal["ResNet18", "VGG11"] = "ResNet18"
    # input channels
    in_channels: int = 3
    # keypoint number
    keypoints: int = 11
    # fully connected units in head hidden layer
    fc_units: int = 4096


@dataclass()
class DataConfig:
    """config for data specification"""

    # input torchvision transform
    transform: Optional[Callable] = ToTensor()
    # target torchvision transform
    target_transform: Optional[Callable] = None


@dataclass()
class TrainConfig:
    """config for training instance"""

    # config for model specification
    model: ModelConfig = field(default_factory=ModelConfig)
    # config for data specification
    data: DataConfig = field(default_factory=DataConfig)
    # run name
    run_name: str = "run_0"
    # experiment name
    exp_name: Optional[str] = None
    # gpu list to expose to training instance
    gpus: List[int] = field(default_factory=lambda: [0, 3])
    # maximum epoch number
    epochs: int = 8
    # batch size
    bs: int = 2
    # enable best checkpoint saving
    early_stop: bool = False
    # number of cpu workers in dataloader
    num_workers: int = 1
    # learning rate
    lr: float = 1e-4
    # weight decay as used in AdamW
    weight_decay: float = 0.1  # only applied on matmul weights
    # AdamW momentum parameters
    betas: tuple = (0.9, 0.95)


pyrallis.decode.register(Callable, lambda x: globals()[x[1:]]())
pyrallis.encode.register(torchvision.transforms.ToTensor, lambda _: "$ToTensor")


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("run_config.yaml", "w"))
