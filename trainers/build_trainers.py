"""Builds the individual components of the trainer,
and the trainer itself.
"""

import math
import os

import torch
from torch.distributed import init_process_group

from models.experimental.hugging_face import MockTrainer
from trainers.base_trainer import BaseTrainer
from trainers.datasets import (
    BaseDataset,
    BaseDatasetRandom,
    BytePoolingDataset,
    DatasetInterface,
    DualBytePooling,
    MultiGPUDataset,
    SingleGPUDataset,
)
from trainers.loss_fn import (
    cross_entropy_loss_fn,
    masked_cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
)
from trainers.optimizer import configure_nanoGPT_optimizer
from trainers.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
    LinearDropoutScheduler,
    LRScheduler,
    TriangleDropoutScheduler,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def ddp_setup(rank, world_size):
    """Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    # Get the master address and port from SLURM environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # Set the environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


OPTIMIZER_DICT = {
    "nanoGPTadamW": lambda model, trainer_cfg: configure_nanoGPT_optimizer(
        model=model,
        weight_decay=trainer_cfg["weight_decay"],
        learning_rate=trainer_cfg["lr"],
        betas=(trainer_cfg["beta1"], trainer_cfg["beta2"]),
    ),
    "adamW": lambda model, trainer_cfg: torch.optim.AdamW(
        model.parameters(),
        lr=trainer_cfg["lr"],
        betas=(trainer_cfg["beta1"], trainer_cfg["beta2"]),
        weight_decay=trainer_cfg["weight_decay"],
    ),
}


def build_optimizer(model, optimizer_config):
    """Given the optimizer config, build the optimizer"""
    return OPTIMIZER_DICT[optimizer_config["name"]](
        model=model, trainer_cfg=optimizer_config
    )


SCHEDULER_DICT = {
    "cosine": lambda trainer_cfg: CosineLRScheduler(
        warmup_iters=trainer_cfg["training"]["warmup_iters"],
        decay_iters=trainer_cfg["training"]["lr_decay_iters"],
        lr=trainer_cfg["optimizer"]["lr"],
        min_lr=trainer_cfg["optimizer"]["min_lr"],
    ),
    "constant": lambda trainer_cfg: LRScheduler(
        lr=trainer_cfg["optimizer"]["lr"],
    ),
}


def build_lr_scheduler(trainer_cfg):
    """Given the trainer config, build the LR scheduler.build_model"""
    return SCHEDULER_DICT[trainer_cfg["lr_scheduler"]["name"]](trainer_cfg=trainer_cfg)


def build_dropout_scheduler(trainer_cfg):
    """Given the trainer config, build the dropout scheduler."""
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "constant":
        return DropoutScheduler(trainer_cfg["dropout_scheduler"]["dropout"])
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "linear":
        return LinearDropoutScheduler(
            start_dropout_p=trainer_cfg["dropout_scheduler"]["start_dropout_p"],
            end_dropout_p=trainer_cfg["dropout_scheduler"]["end_dropout_p"],
            start_iter=trainer_cfg["dropout_scheduler"]["start_iter"],
            end_iter=trainer_cfg["dropout_scheduler"]["end_iter"],
        )
    if trainer_cfg["dropout_scheduler"]["dropout_type"] == "triangle":
        return TriangleDropoutScheduler(
            dropout_trough=trainer_cfg["dropout_scheduler"]["dropout_trough"],
            dropout_peak=trainer_cfg["dropout_scheduler"]["dropout_peak"],
            num_iterations=trainer_cfg["dropout_scheduler"]["num_iterations"],
            num_cycles=trainer_cfg["dropout_scheduler"]["num_cycles"],
        )
    raise NotImplementedError(
        f"dropout scheduler {trainer_cfg['dropout_scheduler']['dropout_type']} not implemented."
    )


DATASET_DICT: dict[str, DatasetInterface] = {
    "normal": BaseDataset,
    "standard": BaseDatasetRandom,
    "single_gpu": SingleGPUDataset,
    "multi_gpu": MultiGPUDataset,
    "byte_pooling": BytePoolingDataset,
    "dual_byte_pooling": DualBytePooling,
}


def build_dataset(cfg, split, seed):
    """Given the config, build the dataloader"""
    return DATASET_DICT[cfg.trainer["dataloader"]["name"]](
        cfg=cfg, split=split, seed=seed
    )


LOSS_FN_DICT = {
    "cross_entropy": cross_entropy_loss_fn,
    "next_token_mlm": next_token_mlm_loss_fn,
    "masked_cross_entropy": masked_cross_entropy_loss_fn,
}


def build_loss_fn(loss_fn_name):
    """Given the loss function name, build the loss function"""
    return LOSS_FN_DICT[loss_fn_name]


TRAINER_DICT = {
    "base_trainer": BaseTrainer,
    "mock_trainer": MockTrainer,
}


def configure_training_parameters(cfg, train_dataset):
    """Configure training parameters including epochs, iterations, and scheduler settings.

    This function handles the calculation of training parameters that were previously
    done in the BaseTrainer constructor. It calculates iterations per epoch, adjusts
    max_epochs/max_iters based on the training mode, and updates scheduler parameters.

    Args:
        cfg: Configuration dictionary containing training parameters
        train_dataset: Training dataset to calculate iterations from

    Returns:
        tuple: (max_epochs, max_iters, is_iters_based, iters_per_epoch, context_window)

    Modifies:
        cfg: Updates lr_decay_iters and warmup_iters in-place based on calculated max_iters
    """
    # Extract training mode configuration
    max_epochs = cfg["trainer"]["training"].get("max_epochs", -1)
    max_iters = cfg["trainer"]["training"].get("max_iters", -1)
    assert max_epochs > 0 or max_iters > 0, (
        "Either max_epochs or max_iters must be positive"
    )
    is_iters_based = True if max_iters > 0 else False

    # Calculate iterations per epoch with error handling
    try:
        context_window = getattr(train_dataset, "context_window", 1)
        iters_per_epoch = math.ceil(
            len(train_dataset)
            / (
                cfg["trainer"]["training"]["batch_size"]
                * cfg["trainer"]["training"]["gradient_accumulation_steps"]
                * context_window
            )
        )
        logger.info(f"Calculated {iters_per_epoch} iterations per epoch")
    except Exception as e:
        logger.warning(f"Could not calculate iters_per_epoch: {e}. Using default.")
        iters_per_epoch = 1000
        context_window = 1

    # Adjust training parameters based on mode
    if is_iters_based:
        max_epochs = max_iters / iters_per_epoch
    else:
        max_iters = max_epochs * math.ceil(iters_per_epoch)

    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"  max_epochs: {max_epochs}")
    logger.info(f"  max_iters: {max_iters}")
    logger.info(f"  is_iters_based: {is_iters_based}")
    logger.info(f"  context_window: {context_window}")
    logger.info(f"  iters_per_epoch: {iters_per_epoch}")

    # Update scheduler parameters in configuration
    cfg.trainer["training"]["lr_decay_iters"] = math.ceil(
        cfg.trainer["training"]["lr_decay_iters"] * max_iters
    )
    logger.info(
        f"  Adjusted lr_decay_iters to {cfg.trainer['training']['lr_decay_iters']} "
        f"based on max_iters"
    )
    cfg.trainer["training"]["warmup_iters"] = math.ceil(
        cfg.trainer["training"]["warmup_iters"] * max_iters
    )
    logger.info(
        f"  Adjusted warmup_iters to {cfg.trainer['training']['warmup_iters']} based on max_iters"
    )

    return max_epochs, max_iters, is_iters_based, iters_per_epoch, context_window


def build_trainer(cfg, model, gpu_id, seed, checkpoint_path=None):
    """Given a config, this function builds a trainer
    and all relevant components of it.

    Args:
        cfg: Configuration dictionary
        model: The model to train
        gpu_id: GPU ID for distributed training
        checkpoint_path: Optional path to checkpoint for resuming training
    """
    logger.info("Building datasets...")
    train_dataset = build_dataset(cfg=cfg, split="train", seed=seed)
    val_dataset = build_dataset(cfg=cfg, split="val", seed=seed)

    # Configure training parameters (moved from BaseTrainer.__init__)
    logger.info("Configuring training parameters...")
    (
        max_epochs,
        max_iters,
        is_iters_based,
        iters_per_epoch,
        context_window,
    ) = configure_training_parameters(cfg, train_dataset)

    logger.info("Building optimizer...")
    optimizer = build_optimizer(model=model, optimizer_config=cfg.trainer["optimizer"])

    logger.info("Building LR scheduler...")
    lr_scheduler = build_lr_scheduler(trainer_cfg=cfg.trainer)

    logger.info("Building dropout scheduler...")
    dropout_scheduler = build_dropout_scheduler(trainer_cfg=cfg.trainer)

    logger.info("Wrapping datasets in dataloaders...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,
        # pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,
        # pin_memory=True,
    )

    logger.info("Building loss function...")
    loss_fn = build_loss_fn(loss_fn_name=cfg.trainer["loss_fn"]["name"])

    logger.info(f"Building trainer of type: {cfg.trainer['training']['trainer_type']}")
    trainer = TRAINER_DICT[cfg.trainer["training"]["trainer_type"]](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        max_epochs=max_epochs,
        max_iters=max_iters,
        is_iters_based=is_iters_based,
        iters_per_epoch=iters_per_epoch,
        dataset_size=len(train_dataset),
        gpu_id=gpu_id,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
    )
    logger.info("Trainer built successfully.")

    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        iteration = trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resuming training from iteration {iteration}")

    return trainer
