"""Utilities used across training repo"""
import os
import json
import shutil
from pathlib import Path
import yaml
import numpy as np
import random
import torch
import collections
from typing import Union


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(y_pred, y):
    """correct predictions / total"""
    return y_pred.eq(y.view_as(y_pred)).float().mean()


def l2(y_pred, y):
    """mean batch l2 norm"""
    return torch.nn.PairwiseDistance(p=2)(y_pred, y).mean()


def zero(y, y_pred):
    """zero criterion"""
    return torch.tensor([0.0], dtype=torch.float32).to(y.device)


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


# May need a better home
def configure_adamw(model, cfg):
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
    for mn, m in model.named_modules():
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
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=cfg.betas)
    return optimizer
