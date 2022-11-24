"""Sample configuration dataclass for training repo."""
from typing import List, Optional, Literal, Union
from dataclasses import dataclass, field
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from pathlib import Path
import pyrallis
from enum import Enum
from dl_schema.models.cfg import ResnetConfig, BabyCNNConfig, VGG11Config
from dl_schema.utils.utils import (
    l2,
    zero,
    accuracy,
    cosine_schedule_with_warmup,
    linear_schedule_with_warmup,
    cosine_with_hard_restarts_schedule_with_warmup,
)

## Wrap, LRMethod, and Criterion are not strictly necessary. These classes implement the
## conveinent ability to ## be able to use callable functions and classes as dataclass
## fields, e.g. `TrainConfig().loss.mse(y_pred, y)`
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
    linear_warmup_cosine_decay: Wrap = Wrap(cosine_schedule_with_warmup)
    linear_warmup_linear_decay: Wrap = Wrap(linear_schedule_with_warmup)
    linear_warmup_cosine_decay_hard_restart: Wrap = Wrap(
        cosine_with_hard_restarts_schedule_with_warmup
    )

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
class DataConfig:
    """config for model specification"""

    # root dir of train dataset
    train_root: Union[str, Path] = "./data/processed/train"
    # root dir of validation dataset
    val_root: Optional[Union[Path, str]] = None
    # root dir of test dataset
    test_root: Optional[Union[Path, str]] = "./data/processed/test"
    # shuffle train dataloader
    shuffle: bool = True
    # drop last batch if it does not divide dataset evenly
    drop_last: bool = True


@dataclass()
class LogConfig:
    """config for logging specification"""

    # mlflow tracking uri
    uri: Optional[str] = "~/dev/dl-schema/dl_schema/mlruns"
    # toggle asynchronous logging (not compatible with ray tune)
    enable_async: bool = True
    # number of threads to use in async logging (2 threads/core typically)
    num_threads: int = 4
    # every `train_freq` steps, log training quantities (metrics, single image batch, etc.)
    train_freq: int = 100
    # every `test_freq` steps, log test quantities (metrics, single image batch, etc.)
    test_freq: int = 500
    # run evaluation before first train step
    evaluate_init: bool = True
    # every `save_freq` steps save model checkpoint according to save criteria
    save_freq: int = 1000
    # save initial model state
    save_init: bool = False
    # save last model state
    save_last: bool = False
    # save best model state (early stopping)
    save_best: bool = False
    # log histograms of trainable parameters
    params: bool = False
    # log histograms of gradients
    gradients: bool = False
    # log images
    images: bool = True


@dataclass()
class TrainConfig:
    """config for training instance"""

    # specification of model class to instantiate
    # (resnet10, resnet12, resnet18, resnet34, resnet50, resnet101, resnet152, VGG11, BabyCNN)
    model_class: str = "resnet18"
    # model config for resnet
    resnet: Optional[ResnetConfig] = field(default_factory=ResnetConfig)
    # model config for vgg11
    vgg11: Optional[VGG11Config] = field(default_factory=VGG11Config)
    # model config for babycnn
    babycnn: Optional[BabyCNNConfig] = field(default_factory=BabyCNNConfig)
    # config for data specification
    data: DataConfig = field(default_factory=DataConfig)
    # config for logging specification
    log: LogConfig = field(default_factory=LogConfig)
    # run name
    run_name: str = "run_0"
    # experiment name
    exp_name: str = "debug"
    # gpu list to expose to training instance
    gpus: List[int] = field(default_factory=lambda: [0])
    # random seed, set to make deterministic
    seed: int = 42
    # number of cpu workers in dataloader
    num_workers: int = 4
    # number of training steps (weight updates)
    train_steps: int = 100
    # batch size
    bs: int = 4
    # learning rate (if onecycle, max_lr)
    lr: float = 1e-4
    # lr schedule type: (constant, onecycle, linear_warmup_cosine_decay,
    # linear_warmup_linear_decay, linear_warmup_cosine_decay_hard_restart)
    lr_method: LRMethod = LRMethod.linear_warmup_cosine_decay
    # initial lr = lr / div
    onecycle_div_factor: float = 25
    # final lr = lr / final_div
    onecycle_final_div_factor: float = 1e4
    # percent of total steps to be in warmup phase
    warmup_pct: int = 10
    # number of hard restart cycles in cosine hard restart lr scheduler
    restart_cycles: int = 2
    # weight decay as used in AdamW
    weight_decay: float = 0.0  # only applied on matmul weights
    # adamw momentum parameters
    betas: tuple = (0.9, 0.95)
    # checkpoint load path
    load_ckpt_pth: Optional[str] = None
    # load optimizer along with weights
    load_optimizer: bool = False
    # resume from last saved epoch in ckpt
    resume: bool = False
    # training loss function : (crossentropy, l1, l2, mse, zero)
    loss: Criterion = Criterion.crossentropy
    # metric function 1: (l1, l2, mse, zero)
    metric1: Criterion = Criterion.accuracy
    # enable ray hyperparam tuning
    tune: bool = False

    def __post_init__(self):
        # set only one model config to be active
        if "resnet" in self.model_class:
            self.babycnn = None
            self.vgg11 = None
        elif "VGG" in self.model_class:
            self.resnet = None
            self.babycnn = None
        else:
            self.resnet = None
            self.vgg11 = None


if __name__ == "__main__":
    """test the train config, export to yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("./configs/train_cfg.yaml", "w"))
