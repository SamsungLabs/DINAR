"""
Functions for merging textures from different view into the single texture.
"""

import os
import pickle
import random
from collections import defaultdict

import torch


def group_by_name(data_dict_list):
    """
    Group paths to data_dicts by their name. Each entry in the resulting dict is a person.

    :param data_dict_list: List of paths to data_dicts
    :return: Dict with paths grouped by persons
    """
    grouped_data_dict = defaultdict(list)
    for filename in data_dict_list:
        splitter = '/'
        person = os.path.splitext(filename.split(splitter)[-1])[0]
        grouped_data_dict[person].append(filename)
    return grouped_data_dict


def merge_textures(main, condition):
    """
    Merge textures for two data_dicts based on normal angles

    :param main: First data_dict (Usually generated by diffusion model)
    :param condition: Second data_dict (Usually obtained from the input image)
    :return: Data_dict based on "main" with a merged texture
    """
    general_data_dict = main

    ntex_list = [main['ntexture'], condition['ntexture']]
    weights_list = [main['normal_angles'], condition['normal_angles']]

    T = 10
    weights_list = torch.cat(weights_list, dim=1) * T
    weights_list = torch.nn.functional.softmax(weights_list, dim=1)

    ntex_list = torch.stack(ntex_list)
    general_data_dict['ntexture'] = torch.einsum("ij...,ji...->j...", ntex_list, weights_list)
    general_data_dict['ntexture'] = general_data_dict['ntexture']

    return general_data_dict


def merge_textures(files_list):
    """
    Load and merge all data_dicts provided by files_list

    :param files_list: List of paths to data_dicts
    :return: Data_dict containing the merged neural texture
    """
    data_dict_list = []
    ntex_list = []
    weights_list = []
    T = 10
    for data_dict_name in files_list:
        with open(data_dict_name, 'rb') as handle:
            data_dict = pickle.load(handle)
            data_dict_list.append(data_dict)

            ntex_list.append(torch.Tensor(data_dict['ntexture']))
            weights_list.append(torch.Tensor(data_dict['normal_angles']) * T)
    weights_list = torch.cat(weights_list, dim=1)
    weights_list = torch.nn.functional.softmax(weights_list, dim=1)

    general_data_dict = random.choice(data_dict_list)
    general_data_dict['ntexture'] = torch.zeros_like(torch.Tensor(general_data_dict['ntexture']))
    for i in range(len(ntex_list)):
        general_data_dict['ntexture'] += ntex_list[i] * weights_list[0, i]
    general_data_dict['ntexture'] = general_data_dict['ntexture'].cpu().detach().numpy()

    return general_data_dict


def find_merged_file(file_list):
    """
    Select precalculated merged file from list of files.
    Select merged file by name

    :param file_list: List of paths to files
    :return: Path to the merged one
    """
    for path in file_list:
        filename = path.split('/')[-1]
        if 'merged' in filename:
            return path
    return None