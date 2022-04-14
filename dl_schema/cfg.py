"""Sample configuration dataclass for training repo."""
from typing import List, Optional, Literal, Union
from dataclasses import dataclass, field
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from pathlib import Path
import pyrallis
from enum import Enum
from dl_schema.utils import l2, zero, accuracy

## Wrap, LRMethod, and Criterion are not strictly necessary. These classes implement the conveinent ability to
## be able to use callable functions and classes as dataclass fields, e.g. `TrainConfig().loss.mse(y_pred, y)`
class Wrap:
    """wrapper for serializing/deserializing classes"""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return repr(self.fn)


class LRMethod(Enum):
    """Enum class for lr methods, used with Wrap"""

    onecycle: Wrap = Wrap(OneCycleLR)
    constant: Wrap = Wrap(LambdaLR)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Criterion(Enum):
    """Enum class for criterion, used with Wrap"""

    mse = Wrap(torch.nn.functional.mse_loss)
    l1 = Wrap(torch.nn.functional.l1_loss)
    l2 = Wrap(l2)
    zero = Wrap(zero)
    crossentropy = Wrap(torch.nn.CrossEntropyLoss())
    accuracy = Wrap(accuracy)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


@dataclass()
class CNNConfig:
    """config for vgg11 specification"""

    # name of model
    model_class: str = "BabyCNN"
    # input image channels
    in_channels: int = 1
    # fully connected units in head hidden layer
    fc_units: int = 128
    # number of output logits in model head
    out_features: int = 10
    # dropout percentage of fc1 layer
    dropout1: float = 0.25
    # dropout percentage of fc2 layer
    dropout2: float = 0.50


@dataclass()
class DataConfig:
    """config for model specification"""

    # root dir of train dataset
    train_tfrecords_path: Optional[Union[str, Path]] = "./data/tfrecords/train.tfrecords"
    # root dir of test dataset
    test_tfrecords_path: Optional[Union[Path, str]] = "./data/tfrecords/test.tfrecords"
    # shuffle dataset
    shuffle: bool = True


@dataclass()
class LogConfig:
    """config for logging specification"""

    # mlflow tracking uri
    uri: Optional[str] = "~/dev/spec21/spec21/keypoint_regression/mlruns"
    # toggle asynchronous logging (experimental)
    enable_async: bool = True
    # frequency to log batch quantities
    batch_freq: int = 1


@dataclass()
class TrainConfig:
    """config for training instance"""

    # config for model specification
    model: CNNConfig = field(default_factory=CNNConfig)
    # config for data specification
    data: DataConfig = field(default_factory=DataConfig)
    # config for logging specification
    log: LogConfig = field(default_factory=LogConfig)
    # run name
    run_name: str = "run_0"
    # experiment name
    exp_name: str = "debug"
    # gpu list to expose to training instance
    gpus: List[int] = field(default_factory=lambda: [-1])
    # random seed, set to make deterministic
    seed: int = 42
    # number of cpu workers in dataloader
    num_workers: int = 4
    # maximum epoch number
    epochs: int = 2
    # batch size
    bs: int = 64
    # learning rate (if onecycle, max_lr)
    lr: float = 3e-4
    # lr schedule type: (constant, onecycle)
    lr_method: LRMethod = LRMethod.onecycle
    # initial lr = lr / div
    onecycle_div_factor: float = 25
    # final lr = lr / final_div
    onecycle_final_div_factor: float = 1e4
    # weight decay as used in AdamW
    weight_decay: float = 0.0  # only applied on matmul weights
    # adamw momentum parameters
    betas: tuple = (0.9, 0.95)
    # save initial model state
    save_init: bool = False
    # save last model state
    save_last: bool = False
    # save best model state (early stopping)
    save_best: bool = False
    # checkpoint load path
    load_ckpt_pth: Optional[str] = None
    # load optimizer along with weights
    load_optimizer: bool = False
    # resume from last saved epoch in ckpt
    resume: bool = False
    # maximum number of steps (overrides epochs)
    steps: Optional[int] = None
    # training loss function : (crossentropy, l1, l2, mse, zero)
    loss: Criterion = Criterion.crossentropy
    # metric function 1: (l1, l2, mse, zero)
    metric1: Criterion = Criterion.accuracy
    # enable ray tune
    tune: bool = False


if __name__ == "__main__":
    """test the train config, export to yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("train_cfg.yaml", "w"))
