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


def quaternion_to_so3(q: list) -> np.ndarray:
    """Convert quaternion to SO(3) representation according to `speedplus-utils`
    convention. Compared to Wikipedia definition, this quantity is transposed
    since q is give in the annotation as q_C, i.e. the camera reference frame.

    Args:
        q: list of shape (4); quaternion (possibly unnormalized)

    Returns:
        so3: np.ndarray of shape (3,3)
    """
    q = q / np.linalg.norm(q)  # normalize
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    so3 = np.zeros((3, 3))

    so3[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    so3[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    so3[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    so3[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    so3[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    so3[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    so3[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    so3[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    so3[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return so3


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_yaml(path: Path):
    """Load yaml as dict.

    Args:
        path: Path to .yaml, .yml, or .json file.
    """
    if Path(path).suffix == ".json":
        return load_from_json(path)

    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_yaml(path, data_dict):
    """Export dict as yaml."""
    # Check if parent directories of path exist; if not, make them.
    Path(path).parent.absolute().mkdir(parents=True, exist_ok=True)

    # Store data (serialize).
    with open(path, "w") as handle:
        yaml.dump(data_dict, handle, default_flow_style=None, sort_keys=False)


def make_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def search_for_files(curr_path: str, file_pattern: str, resolve=False) -> list:
    """Find all paths matching pattern while recurring through directory."""
    file_list = []
    for p in Path(curr_path).rglob(file_pattern):
        if resolve:
            p = p.resolve()
        file_list.append(str(p))
    return file_list


def load_from_json(json_file):
    """
    Pull a detection dictionary into memory from a JSON file.

    :param json_file: path to the JSON file
    :return: dictionary of detections
    """

    with open(json_file, "rb") as handle:
        unserialized_data = json.load(handle)
        handle.close()
        return unserialized_data
