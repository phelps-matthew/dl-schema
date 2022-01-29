"""Utilities used across models in the inference pipeline."""
import os
import json
import shutil
from pathlib import Path
import yaml
import numpy as np
import random
import torch
import collections


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_yaml(path: Path):
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
