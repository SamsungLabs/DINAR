"""
Functions for object instantiation and value and types transformation.
"""

import argparse
import importlib

import numpy as np
import torch


def get_obj_from_str(string, reload=False):
    """
    Return the class selected by a string

    :param string: String with path to the *.py file and class name
    :param reload: Flag to reload object
    :return: Selected class
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Wrapper for object instantiation by it name in the config

    :param config: OmegaConf object with
    :return: Instantiated object with provided parameters
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def str2bool(v):
    """
    Convert string value to bool value

    :param v: String value
    :return: Bool value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_tanh(value):
    """
    Transform values of a tensor to Tanh range [-1, 1]

    :param value: Input tensor in Sigmoid range [0, 1]
    :return: Output tensort in Tanh range [-1, 1]
    """
    return value * 2 - 1.


def to_sigm(value):
    """
    Transform values of a tensor to Sigmoid range [0, 1]

    :param value: Input tensor in Tenh range [-1, 1]
    :return: Output tensor in Sigmoid range [0, 1]
    """
    return (value + 1) / 2


def dict2device(data_dict, device):
    """
    Move all tensors in a dict to the device

    :param data_dict: Dict with tensors
    :param device: Target device
    :return: Pointer to the same dict with all tensors moved on the device
    """
    if hasattr(data_dict, 'to'):
        return data_dict.to(device)

    if type(data_dict) == dict:
        for k, v in data_dict.items():
            data_dict[k] = dict2device(v, device)
    return data_dict


def requires_grad(model, flag=True):
    """
    Modify gradient requires flag for all tensors in the model

    :param model: Model to modify gradient requirement
    :param flag: Value of requires_grad flag
    :return:
    """
    if type(model) == dict:
        for k, v in model.items():
            requires_grad(v, flag)
    elif type(model) == list:
        for k in model:
            requires_grad(k, flag)
    else:
        for p in model.parameters():
            p.requires_grad = flag


def posenc(x, n_coeffs=6, mult=1):
    """
    Positional encoding with periodic functions

    :param x: The position to encode
    :param n_coeffs: Number of coefficients to encode with
    :param mult: Multiplier coefficient
    :return: An encoded value
    """
    rets = [x]
    for i in range(n_coeffs):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2. ** (i * mult) * x))
    return torch.cat(rets, 1)


def make_K(H, W, focal=5000):
    """
    Generate calibration matrix based on the image height and width

    :param H: Image height
    :param W: Image width
    :param focal: Focal length
    :return: Calibration matrix
    """
    K = np.eye(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = W // 2
    K[1, 2] = H // 2
    return K


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def detach_dict(data_dict, to_numpy=False):
    """
    Detach all items in the dict

    :param data_dict: Dict with items to detach
    :param to_numpy: Flag to convert all detached tensors to numpy
    :return:
    """
    for key in data_dict.keys():
        if torch.is_tensor(data_dict[key]):
            if to_numpy:
                data_dict[key] = data_dict[key].cpu().detach().numpy()
            else:
                data_dict[key] = data_dict[key].detach()
        elif isinstance(data_dict[key], dict):
            detach_dict(data_dict[key], to_numpy)