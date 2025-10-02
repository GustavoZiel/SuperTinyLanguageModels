"""A collection of dataloaders"""

import os
import random

import numpy as np
import torch
from torch.distributed import get_rank, get_world_size, is_available, is_initialized
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, get_worker_info

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetInterface(torch.utils.data.IterableDataset):
    """A basic interface to be used by the remaining datasets"""

    def __init__(self, split, cfg):
        """Arguments:
        cfg: the train script cfg
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_name = self.cfg["trainer"]["dataset"]
        self.context_window = self.cfg["model"]["context_window"]
        self.data_path = os.path.join(
            self.cfg["general"]["paths"]["data_dir"],
            self.dataset_name,
            f"{self.cfg['model']['embedder']['tokenizer_type']}-{self.cfg['model']['vocab_size']}-{self.cfg['trainer']['dataloader']['name']}",
            f"{split}.bin",
        )

        self._load_data()
        self.dataset_len = len(self.data) - self.context_window

    def _load_data(self):
        """Get data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist, preprocess the data first")
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
        )
        logger.info(f"Loaded data from {self.data_path}, length: {len(self.data)}")

    def __len__(self):
        """Return dataset length"""
        return self.dataset_len

    def __iter__(self, idx):
        raise NotImplementedError


class BaseDatasetRandom(DatasetInterface):
    """A dataset class that yields random slices of data for training language models.

    This class implements an infinite iterator that, on each iteration, randomly selects a starting index
    and returns a tuple of input and target tensors representing a context window of tokens.

    Args:
        split (str): The dataset split to use (e.g., 'train', 'val', 'test').
        cfg (object): Configuration object containing dataset parameters.

    Methods:
        __iter__():
            Returns an infinite generator that yields (x, y) pairs, where:
                x (torch.Tensor): Input tensor of shape (context_window,) containing token indices.
                y (torch.Tensor): Target tensor of shape (context_window,) containing token indices shifted by one.

    Notes:
        - The data is assumed to be a 1D numpy array of token indices.
        - The context window is defined by self.context_window.
        - The dataset length is defined by self.dataset_len.
        - Each batch is sampled independently and randomly.
    """

    def __init__(self, split, cfg):
        super().__init__(split, cfg)

    def __iter__(self):
        """Get a batch of random data points in an infinite loop."""
        while True:
            # Get a random index
            idx = random.randint(0, self.dataset_len - 1)

            # Extract a slice of data for x and y
            x = torch.from_numpy((self.data[idx : idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy(
                (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
            )

            # Yield the data points
            yield x, y


class BaseDataset(DatasetInterface):
    def __init__(self, split, cfg):
        super().__init__(split, cfg)

        self.worker_info = get_worker_info()
        self.num_workers = self.worker_info.num_workers if self.worker_info is not None else 1
        self.worker_id = self.worker_info.id if self.worker_info is not None else 0

        # Detect if distributed (DDP) is active
        if is_available() and is_initialized():
            self.world_size = get_world_size()
            self.process_rank = get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        if self.world_size > 1:
            logger.info("Using DistributedSampler for BaseDataset")
            num_replicas = self.world_size * self.num_workers
            rank = self.process_rank * self.num_workers + self.worker_id
            self.sampler = DistributedSampler(
                self,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
            )
        else:
            # Not DDP: fall back to a simple random sampler
            logger.info("Using RandomSampler for BaseDataset")
            self.sampler = RandomSampler(self, replacement=False)

    def __iter__(self):
        while True:
            for idx in self.sampler:
                x = torch.from_numpy((self.data[idx : idx + self.context_window]).astype(np.int64))
                y = torch.from_numpy(
                    (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
                )
                yield x, y


class MultiGPUDataset(DatasetInterface):
    def __init__(self, split, cfg):
        super().__init__(split, cfg)

        self.worker_info = get_worker_info()
        self.num_workers = self.worker_info.num_workers if self.worker_info is not None else 1
        self.worker_id = self.worker_info.id if self.worker_info is not None else 0

        self.world_size = get_world_size()
        self.process_rank = get_rank()

        num_replicas = self.world_size * self.num_workers
        rank = self.process_rank * self.num_workers + self.worker_id

        self.sampler = DistributedSampler(
            self,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
        )

    def __iter__(self):
        while True:
            for idx in iter(self.sampler):
                x = torch.from_numpy((self.data[idx : idx + self.context_window]).astype(np.int64))
                y = torch.from_numpy(
                    (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
                )

                yield x, y


class SingleGPUDataset(DatasetInterface):
    def __init__(self, split, cfg):
        super().__init__(split, cfg)
        # self.sampler = SequentialSampler(self)
        self.sampler = RandomSampler(self, replacement=False)

    def __iter__(self):
        for idx in iter(self.sampler):
            # print(f"[DEBUG] SingleGPUDataset __iter__ idx: {idx}")
            # Extract a slice of data for x and y
            x = torch.from_numpy((self.data[idx : idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy(
                (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
            )

            # Yield the data points
            yield x, y


class BytePoolingDataset(DatasetInterface):
    """Simple byte-level dataset"""

    def __init__(self, split, cfg):
        self.loading_shape = None
        super().__init__(split, cfg)
        # force parent init
        self._load_data()

    def _load_data(self):
        """Get data"""
        if self.loading_shape is None:
            data = np.memmap(
                self.data_path,
                dtype=np.uint16,
                mode="r",
            )
            self.loading_shape = (
                len(data) // self.cfg["model"]["embedder"]["byte_context_window"],
                self.cfg["model"]["embedder"]["byte_context_window"],
            )
            data = None
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
            shape=self.loading_shape,
        )

    def __iter__(self):
        """Get a batch of data"""
        while True:
            idx = random.randint(0, self.dataset_len - 1)
            x = torch.from_numpy((self.data[idx : idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy(
                (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
            )
            yield x, y


class DualBytePooling(DatasetInterface):
    """Dataset for both byte-level and higher token level tokens simultaneously"""

    def __init__(self, split, cfg):
        self.loading_shape = None
        # overwrite datapath
        data_folder = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            cfg["trainer"]["dataset"],
            f"{cfg['model']['embedder']['tokenizer_type']}-{cfg['model']['vocab_size']}-{cfg['trainer']['dataloader']['name']}",
        )
        self.data_path_byte = os.path.join(data_folder, f"{split}_byte.bin")
        self.data_path_token = os.path.join(data_folder, f"{split}_token.bin")
        super().__init__(split, cfg)

        # force parent init
        self._load_data()

    def _load_data(self):
        """Get both the byte-level and the token level data"""
        if self.loading_shape is None:
            data = np.memmap(
                self.data_path_byte,
                dtype=np.uint16,
                mode="r",
            )
            self.loading_shape = (
                len(data) // self.cfg["model"]["embedder"]["byte_context_window"],
                self.cfg["model"]["embedder"]["byte_context_window"],
            )
            data = None
        self.data_byte = np.memmap(
            self.data_path_byte,
            dtype=np.uint16,
            mode="r",
            shape=self.loading_shape,
        )
        self.data = np.memmap(
            self.data_path_token,
            dtype=np.uint16,
            mode="r",
        )

    def __iter__(self):
        """Get a batch of data from both the byte and higher token level"""
        while True:
            idx = random.randint(0, self.dataset_len - 1)
            # get byte level batch
            x_byte = torch.from_numpy(
                (self.data_byte[idx : idx + self.context_window]).astype(np.int64)
            )
            # y_byte = torch.from_numpy((self.data_byte[idx + 1: idx + 1 + self.context_window]).astype(np.int64))

            # get token level batch
            # x_token = torch.from_numpy((self.data_token[idx: idx + self.context_window]).astype(np.int64))
            y_token = torch.from_numpy(
                (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
            )
            yield x_byte, y_token
