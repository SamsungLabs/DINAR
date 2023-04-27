"""
Finetune a texture to improve high frequency details.
"""

import argparse
import os
import shutil
import sys
from glob import glob

from omegaconf import OmegaConf
from tqdm import tqdm

import main as finetune_main
from utils.general_utils import str2bool
from utils.merging import group_by_name, find_merged_file


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-d",
        "--data_dict",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="path to pickled data_dict",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-f",
        "--finetune_back",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Flag to finetune colors on the back of an avatar",
    )
    return parser


def finetune(args, grouped_data_dict, out_textures_folder, multiview=False, finetune_back=False):
    """
    Run main.py in finetuning mode for selected data_dicts

    :param args: Arguments list to pass to main.py
    :param grouped_data_dict: data_dicts grouped by names
    :param out_textures_folder: Folder to save resulting texture
    :param multiview: Flag to use merged textures
    :param finetune_back: Flag to adjust colors for the avatar back
    :return:
    """

    pids = sorted(list(grouped_data_dict.keys()))
    for person_id in tqdm(pids):
        files_list = grouped_data_dict[person_id]
        if multiview:
            merged_texture_fn = find_merged_file(files_list)
            files_list = [merged_texture_fn]

        for pretrained_texture_fn in tqdm(files_list):
            print(f"PID: {person_id}, texture filename: {pretrained_texture_fn.split('/')[-1]}")

            if finetune_back:
                max_steps = "128"
                rescale_steps = "64"
            else:
                max_steps = "64"
                rescale_steps = "0"

            # Run finetuning of the texture
            new_argv = args + [
                "--finetune=True",
                "--max_steps=" + str(max_steps),
                "runner.params.rescale_steps=" + rescale_steps,
                "runner.params.pretrained_texture=" + pretrained_texture_fn,
                "train_dataloaders.data.params.frames_subset=[" + person_id +",]",
                "val_dataloaders.data.params.frames_subset=[" + person_id +",]",
            ]
            data_dict_folder = finetune_main.main(new_argv)

            # Find the best by metrics and save
            tuned_data_dict = os.path.join(data_dict_folder, "textures/tuned_data_dict")
            finetuned_texture_fn = sorted(glob(os.path.join(tuned_data_dict, '*')))[-1]
            finetuned_texture_name = finetuned_texture_fn.split('/')[-1]
            shutil.copy(finetuned_texture_fn, os.path.join(out_textures_folder, finetuned_texture_name))


def main(args):
    parser = get_parser()
    opt, unknown = parser.parse_known_args(args)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    data_dict_path = opt.data_dict
    out_textures_folder = '/'.join(data_dict_path.split('/')[:-1] + ["finetuned"])
    if not os.path.exists(out_textures_folder):
        os.makedirs(out_textures_folder)

    data_dict_list = glob(os.path.join(data_dict_path, '*.pkl'))
    grouped_data_dict = group_by_name(data_dict_list)

    finetune(args, grouped_data_dict, out_textures_folder, multiview=config.multiview, finetune_back=opt.finetune_back)

    return str(out_textures_folder)


if __name__ == '__main__':
    main(sys.argv)
