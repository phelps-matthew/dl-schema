"""Utilities used across training repo"""
import collections
import json
import math
from pathlib import Path
import random
from typing import Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def image_grid(x: np.ndarray):
    """Return a single numpy array representing a rendered grid of images in a batch"""
    bs, _, _, _ = x.shape
    # make image grid square
    n_rows = math.ceil(math.sqrt(bs))
    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_rows), axes_pad=0.05)
    for i in range(bs):
        grid[i].imshow(x[i], cmap="gray")
    # remove axes ticks and labels
    plt.setp(grid, xticks=[], yticks=[])
    # render image to canvas
    fig.canvas.draw()
    # save matplotlib figure to np.ndarray
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # reshape from (n) to (3, H, W)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # clear figure and axes for next image
    plt.cla()
    plt.close("all")
    return data


def accuracy(y_pred, y):
    """correct predictions / total"""
    return tf.math.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))


def l2(y_pred, y):
    """mean batch l2 norm"""
    return tf.math.reduce_mean(tf.norm(y_pred-y, ord='euclidean'))


def zero(y, y_pred):
    """zero criterion"""
    return tf.constant(0.0, dtype=tf.float32)


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, step):
        return tf.constant(self.lr)


def flatten(d, parent_key="", sep="."):
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml(path: Union[Path, str]):
    """deserialize yaml as dict

    Args:
        path: Path to .yaml, .yml, or .json file.
    """
    if Path(path).suffix == ".json":
        return load_json(path)

    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_yaml(path, data_dict):
    """serialize dict to yaml."""
    # Check if parent directories of path exist; if not, make them.
    Path(path).parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(data_dict, handle, default_flow_style=None, sort_keys=False)


def load_json(json_file):
    """deserialize json as dict"""
    with open(json_file, "rb") as handle:
        unserialized_data = json.load(handle)
        handle.close()
        return unserialized_data
