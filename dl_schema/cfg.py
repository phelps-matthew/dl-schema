"""Sample configuration dataclass for train experiments"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pyrallis
import tensorflow as tf
import tensorflow_addons as tfa

from dl_schema.utils import ConstantSchedule, accuracy, l2, zero

# NOTE: Wrap, LRMethod, and Criterion are not strictly necessary. These classes implement the conveinent 
# ability to be able to use callable classes and functions as dataclass fields, e.g. `cfg.loss(y_pred, y)`

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

    cyclic: Wrap = Wrap(tfa.optimizers.CyclicalLearningRate)
    constant: Wrap = Wrap(ConstantSchedule)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Criterion(Enum):
    """Enum class for criterion, used with Wrap"""

    mse = Wrap(tf.keras.losses.mse)
    l2 = Wrap(l2)
    zero = Wrap(zero)
    crossentropy = Wrap(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    accuracy = Wrap(accuracy)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


@dataclass()
class CNNConfig:
    """config for babycnn specification"""

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

    # root dir of train dataset (if relative path, relative to dl_schema dir)
    train_tfrecords_path: Optional[Union[str, Path]] = "./data/tfrecords/train.tfrecords"
    # root dir of test dataset (if relative path, relative to dl_schema dir)
    test_tfrecords_path: Optional[Union[Path, str]] = "./data/tfrecords/test.tfrecords"
    # shuffle dataset
    shuffle: bool = True
    # shuffle buffer size
    buffer_size: int = 1024


@dataclass()
class LogConfig:
    """config for logging specification"""

    # toggle asynchronous logging (not supported in dl_schema)
    enable_async: bool = True
    # frequency to log batch quantities
    batch_freq: int = 20
    # toggle logging of input image batch grid
    image_grid: bool = False


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
    # eager mode toggle
    eager: bool = True
    # maximum epoch number
    epochs: int = 2
    # batch size
    bs: int = 64
    # learning rate (if cyclic, max_lr)
    lr: float = 3e-4
    # lr schedule type: (constant, onecycle)
    lr_method: LRMethod = LRMethod.cyclic
    # initial cyclic lr
    cyclic_lr_initial: float = 1e-5
    # number of cycles per epoch (cylic lr)
    cyclic_n_cycles: int = 2
    # weight decay as used in AdamW
    weight_decay: float = 0.0  # only applied on matmul weights
    # adamw momentum beta1
    adam_beta1: float = 0.9
    # adamw momentum beta2
    adam_beta2: float = 0.95
    # save initial model state
    save_init: bool = False
    # save last model state
    save_last: bool = False
    # save best model state (early stopping)
    save_best: bool = False
    # path to weights h5 file
    load_weights_pth: Optional[str] = None
    # path to optimizer npy file
    load_optim_pth: Optional[str] = None
    # maximum number of steps (overrides epochs)
    steps: Optional[int] = None
    # training loss function : (crossentropy, l1, l2, mse, zero)
    loss: Criterion = Criterion.crossentropy
    # metric function 1: (l1, l2, mse, zero)
    metric1: Criterion = Criterion.accuracy


if __name__ == "__main__":
    """test the train config, export to yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("train_cfg.yaml", "w"))
