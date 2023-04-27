"""
Render animation sequence with precalculated avatar.
"""

import argparse
import os.path
import pickle
import sys
from glob import glob

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from utils.visualization.video_maker import VideoMaker


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
        "-a",
        "--animation",
        type=str,
        const=True,
        default="./smplx_data/testseq_azure_amass_merged_poses.pickle",
        nargs="?",
        help="paths to animation sequence",
    )
    parser.add_argument(
        "-z",
        "--zero_threshold",
        type=float,
        const=True,
        default=0.5,
        nargs="?",
        help="Threshold for segmentation masks to make edges more sharp",
    )

    return parser


def make_visualization(opt, video_maker, data_dict, save_path):
    """
    Generate and save visualization video frame by frame

    :param opt: Command line parameters
    :param video_maker: An object of class for video creation
    :param data_dict: Dict structure with necessary information about avatar
    :param save_path: Path to save resulting video file
    :return:
    """
    if opt.animation:
        video_maker.write_seq_animated_video(
            data_dict,
            save_path,
            opt.animation,
            zero_threshold=opt.zero_threshold,
        )
    else:
        video_maker.write_rotation_video(
            data_dict,
            save_path,
            zero_threshold=opt.zero_threshold,
        )


def process_singleview(opt, video_maker, data_dict_list, video_folder):
    """
    Generate videos for all avatars from data_dict_list selected in the slplit

    :param opt: Command line arguments
    :param video_maker: An object of class for video creation
    :param grouped_data_dict: List of data_dicts with the information about avatars
    :param video_folder: Folder to save a video
    :return:
    """

    count = 0
    for data_dict_name in data_dict_list:
        person_id = data_dict_name.split('/')[-1]
        print("Processing", person_id)
        with open(data_dict_name, 'rb') as handle:
            data_dict = pickle.load(handle)

        filename = data_dict_name.split('/')[-1]
        filename = filename.replace('.pkl', '.mp4')
        save_path = os.path.join(video_folder, filename)
        make_visualization(opt, video_maker, data_dict, save_path)
        count += 1
    return count


def main(args):
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args(args)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    video_maker = VideoMaker(
        config.runner.params.first_stage_config.params.renderer_config,
        config.runner.params.first_stage_config.params.rasterizer_config,
        config.runner.params.ckpt_path,
        config.test_dataloader.params.smplx_path,
        config.test_dataloader.params.v_inds_path,
    )

    data_dict_list = sorted(glob(os.path.join(opt.data_dict, '*.pkl')))

    video_folder = os.path.join('/'.join(opt.data_dict.split('/')[:-1]), 'videos')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    process_singleview(opt, video_maker, data_dict_list, video_folder)


if __name__ == '__main__':
    main(sys.argv)
