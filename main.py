"""
Inference one-shot avatar.
"""

import argparse
import datetime
import os.path
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from dataloaders.dataloader_alternating import DataloaderCombiner
from utils.general_utils import instantiate_from_config, str2bool


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
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
        "--finetune",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Flag to finetune a texure",
    )

    return parser


def create_dataloaders(config, create_test_dataloader=False):
    """
    Create Train, Test and Validation dataloaders based on config parameters

    :param config: OmegaConf config with dataloader parameters
    :param create_test_dataloader: Flag to create Test dataloader (during training stage we need only Train and Val)
    :return: Three objects with Train, Val and Test dataloaders. If dataloader not created returns None.
    """
    train_dataloader = None
    val_dataloaders = None
    test_dataloader = None

    if config.get("train_dataloaders", None):
        train_dataloader_combiner = DataloaderCombiner()
        for dataset_config in config.train_dataloaders.values():
            if type(dataset_config) != DictConfig:
                continue
            dataset = instantiate_from_config(dataset_config)
            train_dataloader_combiner.add_dataloader(dataset, dataset_config.probability)

        train_dataloader = train_dataloader_combiner.combined_dataloader(
            config.train_dataloaders.batch_size,
            config.train_dataloaders.num_workers,
        )

    if config.get("val_dataloaders", None):
        dataloaders = dict()
        for dataset_name, dataset_config in config.val_dataloaders.items():
            if type(dataset_config) != DictConfig:
                continue
            val_dataset = instantiate_from_config(dataset_config)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.val_dataloaders.batch_size,
                num_workers=config.val_dataloaders.num_workers,
                shuffle=False,
                pin_memory=False,
                drop_last=True,
            )
            dataloaders[dataset_name] = val_dataloader

        val_dataloaders = [CombinedLoader(dataloaders)]
        if config.val_dataloaders.double_val_dataloader:
            val_dataloaders.append(CombinedLoader(dataloaders))

    if config.get("test_dataloader", None) and create_test_dataloader:
        test_dataset = instantiate_from_config(config.test_dataloader)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.test_dataloader.batch_size,
            num_workers=config.test_dataloader.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=True
        )

    return train_dataloader, val_dataloaders, test_dataloader


def create_callbacks(config):
    """
    Instantiate the list of callbacks from the config file

    :param config: OmegaConf object with callbacks parameters
    :return: List of instantiated callbacks
    """
    callbacks = []
    for callback_conf in config.callbacks.values():
        callbacks.append(instantiate_from_config(callback_conf))

    return callbacks


def generate_path_to_logs(config, opt):
    """
    Generate path to output folder with experiment logging.
    The path contains config name and launch datetime

    :param config: OmegaConf object contain base folder for experiments
    :param opt: Command line parameters with config file name and launch mode
    :return: Path to directory with experiments
    """
    experiment_name = opt.base[0].split('/')[-1].split('.yaml')[0]
    time = datetime.datetime.now()
    run_name = time.strftime(f"run_%Y_%m-%d_%H-%M")
    if not opt.finetune:
        run_name += '_test'
    if opt.finetune:
        log_dir = os.path.join(
            config.logdir, experiment_name, '_'.join(config.train_dataloaders.data.params.frames_subset), run_name
        )
    else:
        log_dir = os.path.join(config.logdir, experiment_name, run_name)
    return log_dir


def main(args):
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args(args)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    runner = instantiate_from_config(config.runner)

    # Adjust learning rate by batch size and gpu number
    if "base_learning_rate" in config.runner:
        base_lr = config.runner.base_learning_rate
        if config.get("train_dataloaders", None):
            bs = config.train_dataloaders.batch_size
        else:
            bs = 0
        ngpu = config.gpus
        runner.learning_rate = ngpu * bs * base_lr

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        config,
        create_test_dataloader=not opt.finetune,
    )

    log_dir = generate_path_to_logs(config, opt)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, flush_secs=config.flush_log_secs)

    # Training
    trainer = pl.Trainer(
        max_epochs=opt.max_epochs,
        max_steps=opt.max_steps,
        accelerator=config.accelerator,
        gpus=config.gpus,
        logger=tb_logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        callbacks=create_callbacks(config),
    )

    if opt.finetune:
        trainer.fit(runner, train_dataloader, val_dataloader)
    else:
        trainer.test(runner, test_dataloader)

    return log_dir


if __name__ == '__main__':
    main(sys.argv)
