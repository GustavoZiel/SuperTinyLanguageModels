"""Utilities for the trainer"""

import importlib
import inspect
import math
import os
import pkgutil
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torch.distributed as dist
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from prettytable import PrettyTable
from torch.utils.data import IterableDataset

from utils.logger import get_logger

logger = get_logger()

DATA_DIR = "/home/ziel/codes/SIPGA/scripts/data"


class IterableDatasetShard(IterableDataset):
    """Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:

    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`

    <Tip warning={true}>

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a `torch.Generator` in a `generator` attribute of the `dataset` to
        generate your random numbers and call the [`~trainer_pt_utils.IterableDatasetShard.set_epoch`] method of this
        object. It will set the seed of this `generator` to `seed + epoch` on all processes before starting the
        iteration. Alternatively, you can also implement a `set_epoch()` method in your iterable dataset to deal with
        this.

    </Tip>

    Args:
        dataset (`torch.utils.data.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard.
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_pt_utils.IterableDatasetShard.set_epoch`].
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(
            self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size
        )

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]

    def __len__(self):
        # Will raise an error if the underlying dataset is not sized.
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return (
                math.ceil(len(self.dataset) / (self.batch_size * self.num_processes))
                * self.batch_size
            )


def load_custom_dataset(dataset_name: str) -> Any:
    return load_from_disk(os.path.join(DATA_DIR, dataset_name))


def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def get_absolute_path(relative_path: str, verbose: bool = False) -> str:
    """Get the absolute path from a relative path.

    Args:
        relative_path (str): The relative path to convert.
        verbose (bool, optional): If True, logs the absolute path. Defaults to False.

    Returns:
        str: The absolute path.
    """
    absolute_path = hydra.utils.to_absolute_path(relative_path)
    if verbose:
        logger.info(f"'{relative_path}' directory set to: {absolute_path}")
    return absolute_path


def create_folder(relative_path: str, verbose: bool = False) -> None:
    """Create a folder if it does not exist.

    Args:
        relative_path (str): The relative path of the folder to create.
        verbose (bool, optional): If True, logs folder creation. Defaults to False.
    """
    absolute_path = get_absolute_path(relative_path, verbose=verbose)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
        if verbose:
            logger.info(f"Created folder: {absolute_path}")


def create_folder_structure(*args, verbose: bool = False) -> None:
    """Create all necessary folders for training.

    Args:
        *args: Relative paths of folders to create.
        verbose (bool, optional): If True, logs folder creation. Defaults to False.
    """
    for path in args:
        create_folder(path, verbose=verbose)


def create_stlm_data_mix():
    """A small custom datamix for STLM models containing:
    - simple English Wikipedia
    - Python Code (Deepmind Code Contest) - sampled for easy questions
    - technical QA style (StackExchange)
    """
    # Load simple English Wikipedia
    wiki = load_dataset("wikimedia/wikipedia", "20231101.simple")["train"]

    # Add a "text" column for simple English Wikipedia
    wiki = wiki.map(lambda x: {"text": x["text"]})

    # Load Python code from DeepMind Code Contests
    code_dataset = load_dataset("jtatman/python-code-dataset-500k")["train"]
    code_dataset = code_dataset.map(
        lambda x: {"text": f"Instruction: {x['instruction']}\nOutput: {x['output']}"}
    )

    # Load technical QA style data from StackExchange
    openhermes = load_dataset("teknium/OpenHermes-2.5")["train"]

    # Transform to have a "text" column with both question and answers
    openhermes = openhermes.map(
        lambda x: {
            "text": f"Question: {x['conversations'][0]['value']}\nAnswers: {x['conversations'][1]['value']}"
        }
    )

    # Add tiny stories
    tiny_stories = load_dataset("roneneldan/TinyStories")["train"]

    # Calculate and print the distribution of string lengths
    def calculate_length_distribution(dataset):
        lengths = [len(item["text"]) for item in dataset]
        return sum(lengths), lengths

    wiki_length, wiki_lengths = calculate_length_distribution(wiki)
    python3_code_length, python3_code_lengths = calculate_length_distribution(code_dataset)
    openhermes_length, openhermes_lengths = calculate_length_distribution(openhermes)
    tiny_stories_length, tiny_stories_lengths = calculate_length_distribution(tiny_stories)

    total_length = wiki_length + python3_code_length + openhermes_length + tiny_stories_length

    print(f"Wiki Text Length: {wiki_length} ({wiki_length / total_length * 100:.2f}%)")
    print(
        f"Python Code Text Length: {python3_code_length} ({python3_code_length / total_length * 100:.2f}%)"
    )
    print(
        f"openhermes Text Length: {openhermes_length} ({openhermes_length / total_length * 100:.2f}%)"
    )

    # Concatenate datasets
    combined_dataset = concatenate_datasets([wiki, code_dataset, openhermes, tiny_stories])

    combined_dataset = DatasetDict(
        {
            "train": combined_dataset,
        }
    )

    return combined_dataset


def load_github_code_dataset():
    """Load and re-format the github code dataset
    https://huggingface.co/datasets/codeparrot/github-code
    """
    dataset = load_dataset("codeparrot/github-code")

    # rename "code" column to "text" column
    dataset = dataset.map(lambda x: {"text": x["code"]})["train"]

    # dataset = DatasetDict({
    #    "train": dataset,
    # })

    return dataset


def load_competition_math_dataset():
    """Load and re-format the competition math dataset
    https://huggingface.co/datasets/hendrycks/competition_math
    """
    dataset = load_dataset("hendrycks/competition_math")

    # format the problem and solution into a single "text" column
    dataset = dataset.map(lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"})

    dataset = DatasetDict(
        {
            "train": dataset,
        }
    )

    return dataset


DATASET_DICT = {
    "sports_wiki": lambda: load_custom_dataset("wiki_20220301.en_filtered"),
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset("Sree1994/babylm_100M"),  # https://babylm.github.io/
    "tinystories": lambda: load_dataset(
        "roneneldan/TinyStories"
    ),  # https://huggingface.co/datasets/roneneldan/TinyStories
    "stlm": create_stlm_data_mix,
    "openhermes-2.5": lambda: load_dataset("teknium/OpenHermes-2.5"),
    "openwebtext": lambda: load_dataset("Skylion007/openwebtext"),
    "github-code": lambda: load_github_code_dataset(),
    "competition_math": lambda: load_competition_math_dataset(),
}


def load_data(
    dataset_name: str,
    test_size: float = 0.01,
    seed: int = 489,
    shuffle: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load a dataset by name, split it into training and validation sets, and optionally shuffle and log details.

    Args:
        dataset_name (str): The name of the dataset to load. Must be a key in DATASET_DICT.
        test_size (float, optional): Proportion of the dataset to use as validation. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 489.
        shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.
        verbose (bool, optional): If True, logs dataset loading and split information. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary with "train" and "val" splits of the dataset.

    Raises:
        AssertionError: If the dataset_name is not found in DATASET_DICT.
    """
    assert dataset_name in DATASET_DICT, f"Dataset {dataset_name} not found!"
    if verbose:
        logger.info(f"Loading dataset: {dataset_name}")
    dataset = DATASET_DICT[dataset_name]()

    if verbose:
        logger.info(f"Splitting dataset: test_size={test_size}, seed={seed}, shuffle={shuffle}")
    split_dataset = dataset["train"].train_test_split(
        test_size=test_size, seed=seed, shuffle=shuffle
    )

    split_dataset["val"] = split_dataset.pop("test")

    if dataset_name == "debug":
        split_dataset["train"] = split_dataset["train"].select(range(2048))
        if verbose:
            logger.info("Debug mode: selected first 2048 samples for training.")

    if verbose:
        logger.info(
            f"Dataset '{dataset_name}' loaded. Train size: {len(split_dataset['train'])}, "
            f"Val size: {len(split_dataset['val'])}"
        )

    return split_dataset


def get_classes_from_module(module_name):
    """Get a list of classes defined in a module or package.

    Args:
        module_name (str): The name of the module or package.

    Returns:
        list: A list of classes defined in the module or package.
    """
    module = importlib.import_module(module_name)
    classes = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if inspect.getmodule(obj) == module:
            classes.append(obj)

    return classes


def get_classes_from_package(package_name):
    """Get a list of classes defined in a package and its subpackages.

    Args:
        package_name (str): The name of the package.

    Returns:
        list: A list of classes defined in the package and its subpackages.
    """
    package = importlib.import_module(package_name)
    classes = get_classes_from_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        classes.extend(get_classes_from_module(module_name))

    return classes


def register_backward_hooks(tensor, module_name):
    """Registers hooks to profile the backward pass of a tensor."""
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

        def backward_hook(grad):
            with torch.autograd.profiler.record_function(f"{module_name}.backward"):
                return grad

        tensor.register_hook(backward_hook)


def profilize(model, classes=None):
    """Recursively add hooks to the model for recording PyTorch profiler traces with module names"""
    if classes is None:
        classes = get_classes_from_package("models")
        classes += get_classes_from_package("models.components.layers")
        print(f"Found classes for profiling: {classes}")

    for module in model.children():
        if isinstance(module, torch.nn.Module):
            profilize(module, classes=classes)
        if isinstance(module, torch.nn.ModuleDict):
            for sub_module in module.values():
                profilize(sub_module, classes=classes)
        if isinstance(module, torch.nn.ModuleList):
            for sub_module in module:
                profilize(sub_module, classes=classes)

    if (
        hasattr(model, "forward")
        and any(isinstance(model, cls) for cls in classes)
        and not hasattr(model, "old_forward")
    ):
        model.old_forward = model.forward
        print(f"added forward profiling wrapper for {model.__class__.__name__}")

        def forward_wrapper(*args, **kwargs):
            nested_module_name = model.__class__.__name__
            with torch.autograd.profiler.record_function(f"{nested_module_name}.forward"):
                outputs = model.old_forward(*args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    register_backward_hooks(output, nested_module_name)
            else:
                register_backward_hooks(outputs, nested_module_name)
            return outputs

        model.forward = forward_wrapper


def is_dist():
    """Check if the current process is distributed."""
    return dist.is_initialized()


def aggregate_value(value, device=torch.device("cuda")):
    """Since using DDP, calculation of metrics happen across all GPUs.
    This function aggregate the loss across all GPUs.
    """
    if not is_dist():
        return value
    all_loss = torch.tensor([value], device=device)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    return all_loss.item() / dist.get_world_size()
    # return value


def init_logger_override(logger):
    """Override logger methods so only rank 0 logs to the console.
    Returns a dict of the original methods so you can restore if needed.
    """
    original_methods = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical,
        "exception": logger.exception,
    }

    def make_wrapper(original_method):
        def wrapper(*args, **kwargs):
            if os.getenv("GLOBAL_RANK", "0") == "0":
                original_method(*args, **kwargs)

        return wrapper

    # Override logger methods
    logger.debug = make_wrapper(logger.debug)
    logger.info = make_wrapper(logger.info)
    logger.warning = make_wrapper(logger.warning)
    logger.error = make_wrapper(logger.error)
    logger.critical = make_wrapper(logger.critical)
    logger.exception = make_wrapper(logger.exception)

    return original_methods


def restore_logger_override(logger, original_methods):
    """Restore the original logger methods after overriding.

    Args:
        logger (logging.Logger): The logger to restore.
        original_methods (dict): Dict returned by init_logger_override.
    """
    for method_name, original_method in original_methods.items():
        setattr(logger, method_name, original_method)


def init_print_override():
    """Overriding the print function is useful when running DDP.
    This way, only rank 0 prints to the console.
    """
    import builtins as __builtin__

    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv("GLOBAL_RANK") == "0":
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print


def restore_print_override(original_print):
    """Restore the original print function."""
    import builtins as __builtin__

    __builtin__.print = original_print


# Function to print evaluation results and benchmark results
def print_evaluation_results(iter_num, eval_results, benchmark_results):
    headers = ["Metric", "Value"]
    table = PrettyTable(headers)

    # Adding eval_results rows
    for metric, value in eval_results.items():
        row = [metric, value]
        table.add_row(row)

    print(f"Iteration {iter_num}")
    print(table)

    benchmark_table = PrettyTable(["Benchmark", "Accuracy", "Path Conf.", "Ground Conf."])
    for eval_method in benchmark_results.keys():
        if eval_method == "ft_qa":
            continue
        for benchmark, value in benchmark_results[eval_method].items():
            benchmark_table.add_row(
                [
                    f"{benchmark}",
                    value["accuracy"],
                    value["path_confidence"],
                    value["ground_confidence"],
                ]
            )

    print("Benchmark Results")
    print(benchmark_table)
