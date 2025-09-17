"""Necessary to be run before training to make sure all of the data is preprcessed etc."""

import os

import numpy as np
from tqdm import tqdm

from models.build_models import build_embedding_model
from trainers.utils import load_data
from utils.logger import get_logger

logger = get_logger()


class StandardProcessor:
    """A standard processor that tokenizes the text"""

    def __init__(self, embedder):
        self.embedder = embedder

    def process(self, example):
        ids = self.embedder.tokenize_input(example["text"])
        return {"ids": ids, "len": len(ids)}

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """Write the tokenized data to a file"""
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    def write_tokenized_data_easy(
        self, tokenized, tokenized_data_folder, dtype=np.uint16, total_batches=None
    ):
        """Write tokenized datasets to disk as flat binary files using memmap.

        Args:
            tokenized (dict): Dictionary of Hugging Face Datasets (split -> Dataset)
            tokenized_data_folder (str): Folder to save .bin files
            dtype (np.dtype): data type for token IDs (default uint16)
            total_batches (int, optional): Number of shards to split dataset into. If None, auto-determined.
        """
        for split, dset in tokenized.items():
            arr_len = int(np.sum(dset["len"], dtype=np.uint64))
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")

            # Use provided total_batches or auto-adapt to dataset size
            num_batches = total_batches or min(1024, len(dset))
            print(
                f"Writing split '{split}' to {filename} ({arr_len} tokens in {num_batches} batches)"
            )

            # Create a flat memmap array
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

            idx = 0
            for batch_idx in tqdm(range(num_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=num_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")

                # Flatten token IDs for this batch
                arr_batch = np.concatenate(batch["ids"])

                # Safety check: prevent writing past allocated size
                end_idx = idx + len(arr_batch)
                if end_idx > arr.shape[0]:
                    raise ValueError(
                        f"Batch exceeds preallocated array size: {end_idx} > {arr.shape[0]}"
                    )

                arr[idx:end_idx] = arr_batch
                idx = end_idx

            arr.flush()
            print(f"✅ Finished writing {split}, total tokens: {idx}")

    def write_tokenized_data_simpler(tokenized, tokenized_data_folder, dtype=np.uint16):
        for split, dset in tokenized.items():
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            all_ids = np.concatenate(dset["ids"])
            all_ids = all_ids.astype(dtype)
            all_ids.tofile(filename)  # simple flat write
            print(f"Saved {split} ({len(all_ids)} tokens) to {filename}")


class ByteLevelProcessor(StandardProcessor):
    """A byte-level processor that tokenizes the text"""

    def __init__(self, embedder):
        super().__init__(embedder)

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(
                filename,
                dtype=dtype,
                mode="w+",
                shape=(arr_len, 12),  # TODO remove hardcoding
            )
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()


class DualByteLevelProcessor(StandardProcessor):
    """This preprocessor stores both the byte level structure and
    the standard structure to enable the training of architectures
    with byte-level input, but standard token output.
    """

    def __init__(self, embedder):
        super().__init__(embedder)

    def process(self, example):
        byte_ids, token_ids = self.embedder.tokenize_input(example["text"], return_high_level=True)
        return {"byte_ids": byte_ids, "token_ids": token_ids, "len": len(token_ids)}

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)

            filename_byte = os.path.join(tokenized_data_folder, f"{split}_byte.bin")
            filename_token = os.path.join(tokenized_data_folder, f"{split}_token.bin")

            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

            arr_byte = np.memmap(
                filename_byte,
                dtype=dtype,
                mode="w+",
                shape=(arr_len, 12),  # TODO remove hardcoding
            )

            arr_token = np.memmap(
                filename_token,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )

            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(
                range(total_batches), desc=f"writing {filename_byte} and {filename_token}"
            ):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch_byte = np.concatenate(batch["byte_ids"])
                arr_batch_token = np.concatenate(batch["token_ids"])

                # write into mmap
                arr_byte[idx : idx + len(arr_batch_byte)] = arr_batch_byte
                arr_token[idx : idx + len(arr_batch_token)] = arr_batch_token
                idx += len(arr_batch_byte)

            arr_byte.flush()
            arr_token.flush()


DATALOADER_PROCESSORS = {
    "standard": StandardProcessor,
    "byte_pooling": ByteLevelProcessor,
    "dual_byte_pooling": DualByteLevelProcessor,
}


def create_tokenized_data_folder(cfg, verbose=True):
    """Create the folder to store the tokenized data"""
    logger.info("Creating tokenized data folder")
    dataset_name = cfg["trainer"]["dataset"]
    tokenized_data_folder = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        dataset_name,
        f"{cfg['model']['embedder']['tokenizer_type']}-{cfg['model']['vocab_size']}-{cfg['trainer']['dataloader']['name']}",
    )
    if not os.path.exists(tokenized_data_folder):
        os.makedirs(tokenized_data_folder)
        if verbose:
            logger.info(f"Created tokenized data folder at {tokenized_data_folder}")
    else:
        if verbose:
            logger.info(f"Tokenized data folder already exists at {tokenized_data_folder}")


def prepare_data(cfg):
    """Split the data, process & tokenize it, and store
    it as memmap bin files
    """
    create_tokenized_data_folder(cfg, verbose=True)

    # load embedder
    embedder = build_embedding_model(cfg["model"], verbose=True)

    # # load the dataset
    # dataset_name = cfg["trainer"]["dataset"]
    # logger.info(f"Loading dataset: {dataset_name}")
    # split_dataset = load_data(
    #     dataset_name=dataset_name,
    # )

    # dataloader_name = cfg["trainer"]["dataloader"]["name"]
    # processor_object = DATALOADER_PROCESSORS[dataloader_name](embedder=embedder)
    # logger.info(f"Using processor: {processor_object.__class__.__name__}")

    # # wrap in try such that half-complete files can be deleted on error
    # try:
    #     # Get the maximum number of processors
    #     max_procs = os.cpu_count()
    #     # cap at 12 to reduce memory usage
    #     # max_procs = 1  # min(max_procs, 12) # TODO properly fix this
    #     max_procs = min(max_procs, 12)  # TODO properly fix this
    #     logger.info(f"Using {max_procs} processors for tokenization")

    #     # tokenize the dataset
    #     logger.info("Tokenizing dataset")
    #     tokenized = split_dataset.map(
    #         processor_object.process,
    #         remove_columns=["text"],
    #         desc="Tokenizing dataset",
    #         num_proc=max_procs,
    #     )

    #     # concatenate all the ids in each dataset
    #     logger.info(f"Writing tokenized data to {tokenized_data_folder}")
    #     # processor_object.write_tokenized_data(
    #     processor_object.write_tokenized_data_easy(
    #         tokenized=tokenized, tokenized_data_folder=tokenized_data_folder
    #     )
    #     logger.info("Tokenized data successfully written")

    # except Exception as exc:
    #     logger.error(f"Error during data preparation: {exc}")
    #     for file in os.listdir(tokenized_data_folder):
    #         os.remove(os.path.join(tokenized_data_folder, file))
    #     raise RuntimeError("Failed to process and write data") from exc
