import os
import random
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import tyro
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from pydantic import BaseModel, Field
from torch.distributed import get_rank, get_world_size
from typing_extensions import Literal

DATA_PATH = "/home/ziel/codes/SIPGA/scripts/data"


def load_custom_dataset(dataset_name: str):
    return load_from_disk(os.path.join(DATA_PATH, dataset_name))


DATASET_DICT = {
    "sports_wiki": lambda: load_custom_dataset("wiki_20231101.en_filtered"),
    # ---
    "wiki_biology": lambda: load_dataset("mattany/wikipedia-biology"),
    "wiki_movies": lambda: load_dataset("yashassnadig/wikimovies"),
    "nano_wiki": lambda: load_dataset("sixf0ur/nano_wiki"),
    "wiki_paragraphs": lambda: load_dataset("agentlans/wikipedia-paragraphs"),
    "wiki_solarsystem": lambda: load_dataset("mattany/wikipedia-solarsystem"),
    "wiki_3000": lambda: load_dataset("not-lain/wikipedia-small-3000-embedded"),
    "ap_news_2024": lambda: load_dataset("PJMixers/AP-News-2024"),
    # ---
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"),  # https://babylm.github.io/
    "tinystories": lambda: load_dataset(
        "roneneldan/TinyStories"
    ),  # https://huggingface.co/datasets/roneneldan/TinyStories
}


class Config(BaseModel):
    src_dataset: Literal[tuple(DATASET_DICT.keys())] = Field(
        ...,
        description="Name of the dataset",
    )
    dst_dataset: str = Field(
        ...,
        description="Name of the new dataset to save",
    )
    split_name: Literal["train", "val", "test"] = Field(
        "train",
        description="Data split to use",
    )
    seed: int = Field(42, description="Random seed")
    new_text: List[str] = Field(
        ...,
        description="List of new textes to insert into the dataset",
    )


def main(config: Config):
    """Main function to insert a new text row into a specified split of a HuggingFace dataset,
    then save the modified dataset to disk.

    Args:
        config (Config): Configuration object containing dataset names, split, and new text.
    """
    print("Loading dataset with config:", config)

    # Load the source dataset using the provided name
    df = DATASET_DICT[config.src_dataset]()

    # Check if the specified split exists
    if config.split_name not in df.keys():
        raise ValueError(
            f"'{config.split_name}' not present in dataset splits: {df.keys()}, provide a valid split name"
        )

    # Check if the split contains a 'text' field
    if "text" not in list(df[config.split_name].features.keys()):
        raise ValueError("Dataset does not have a 'text' field")

    print(f"Original dataset: {df}")

    new_rows_qtt = len(config.new_text)
    new_rows = Dataset.from_dict(
        {
            k: [""] * new_rows_qtt if k != "text" else config.new_text
            for k in df[config.split_name].features
        }
    )

    print("New row to insert:", new_rows)
    for i in range(new_rows_qtt):
        print(f"New row {i} content:", new_rows[i])

    # Concatenate the new row to the specified split
    df[config.split_name] = concatenate_datasets([df[config.split_name], new_rows])
    print("New dataset:", df)

    # Save the modified dataset to disk
    save_path = os.path.join(DATA_PATH, config.dst_dataset)
    print("Saving new dataset to", save_path)

    df.save_to_disk(save_path)
    print("Dataset saved successfully.")

    # Optionally, reload and print confirmation
    read_dataset = DatasetDict.load_from_disk(save_path)
    print("Reloaded new dataset:", read_dataset)
    for i in range(new_rows_qtt):
        print(f"Reloaded new row {i} content:", read_dataset[config.split_name][-new_rows_qtt + i])


if __name__ == "__main__":
    config = tyro.cli(Config)
    assert isinstance(config, Config)
    main(config)
