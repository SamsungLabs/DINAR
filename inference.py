"""
End-2-end pipeline for texture generation, finetuning and visualization.
"""

import argparse
import os.path

import finetune_texture
import main
import visualize


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-с",
        "--ckpt_path",
        type=str,
        const=True,
        default="./checkpoints/last-epoch=24-step=46824.ckpt",
        nargs="?",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        const=True,
        default="./datasets/people_snapshot/",
        nargs="?",
        help="Path to folder with data to process",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        const=True,
        default="./logs/",
        nargs="?",
        help="Path to folder to save resulting avatars",
    )
    parser.add_argument(
        "-f",
        "--finetune_back",
        type=str,
        const=True,
        default="False",
        nargs="?",
        help="Flag to finetune colors on the back of an avatar",
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # -------------------------------------------
    argv = [
        "",
        "--base=./configs/eval.yaml",
        "test_dataloader.params.data_root=" + args.data_root,
        "runner.params.ckpt_path=" + args.ckpt_path,
        "logdir=" + args.log_dir,
    ]
    data_dict_folder = main.main(argv)
    data_dict_folder = os.path.join(data_dict_folder, 'textures/data_dict')

    # -------------------------------------------

    argv = [
        "",
        "--base=./configs/finetune_avatar.yaml",
        "--data_dict=" + data_dict_folder,
        "--finetune_back=" + args.finetune_back,
        "train_dataloaders.data.params.data_root=" + args.data_root,
        "val_dataloaders.data.params.data_root=" + args.data_root,
        "runner.params.ckpt_path=" + args.ckpt_path,
        "logdir=" + args.log_dir,
    ]
    finetuned_folder = finetune_texture.main(argv)

    # -------------------------------------------

    argv = [
        "",
        "--base=./configs/eval.yaml",
        "--data_dict=" + finetuned_folder,
        "runner.params.ckpt_path=" + args.ckpt_path,
        "logdir=" + args.log_dir,
    ]
    visualize.main(argv)

    # -------------------------------------------
