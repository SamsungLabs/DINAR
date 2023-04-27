import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def worker_init_fn(_):
    """
    Function to initialize random seeds for different workers.
    I'm not shure if we need this but at this point to afraid to remove
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class AlternatingDataloader:
    """
    Dataloader that stores several dataloaders and sample the random one on each step
    """
    def __init__(self):
        """
        Create an empty list of dataloaders and probabilities to sample from them
        """
        self.dataloader_list = []
        self.probability_list = []

    def add_dataloader(self, dataloader, probability):
        """
        Add a dataloader

        :param dataloader: Pointer to the dataloader
        :param probability: Probability to sample from it
        :return:
        """
        self.dataloader_list.append(dataloader)
        self.probability_list.append(probability)

    def __len__(self):
        """
        Number of iteration steps is equal to minimum length of provided dataloaders
        :return: Number of iterations
        """
        lens = [(len(dataloader) - 1) for dataloader in self.dataloader_list]
        return min(lens)

    def __iter__(self):
        """
        Sample from random dataloader with provided probabilities
        :return:
        """
        iterators = []
        for dataloader in self.dataloader_list:
            iterators.append(dataloader.__iter__())

        while True:
            iterator = random.choices(iterators, weights=self.probability_list)
            data_dict = next(iterator[0], None)

            yield data_dict


class DataloaderCombiner:
    """
    Wrapper for the combined dataloader above
    """
    def __init__(self):
        """
        Create an empty list of dataloaders and probabilities to sample from them
        """
        self.dataset_list = []
        self.probability_list = []

    def add_dataloader(self, dataset, probability):
        """
        Add a dataloader

        :param dataloader: Pointer to the dataloader
        :param probability: Probability to sample from it
        :return:
        """
        self.dataset_list.append(dataset)
        self.probability_list.append(probability)

    def combined_dataloader(
            self,
            batch_size,
            num_workers,
    ):
        """
        Creates alternating dataloader.

        :param batch_size: Batch size. All data into the batch from the same dataloader
        :param num_workers: Number of workers to process dataloader
        :return: AlternatingDataloader instance with datasets added with add_dataloader
        """
        combined = AlternatingDataloader()
        for dataset, probability in zip(self.dataset_list, self.probability_list):
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
                worker_init_fn=worker_init_fn,
            )

            combined.add_dataloader(dataloader, probability)

        return combined
