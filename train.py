"""The main training code"""

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from utils.logger import get_logger

logger = get_logger()

import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from models.build_models import build_model
from models.utils import print_model_stats
from trainers import base_trainer
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.prepare import prepare_data
from trainers.utils import (
    create_folder_structure,
    init_logger_override,
    init_print_override,
    restore_logger_override,
    restore_print_override,
)


def ddp_main(rank, world_size, cfg):
    """Main function for distributed training"""
    os.environ["GLOBAL_RANK"] = str(rank)

    # override the print function to include rank info
    original_print = init_print_override()

    # override the logger to include rank info
    # originals = init_logger_override(logger)

    try:
        # print("Rank: ", rank, "World Size: ", world_size)
        logger.info(f"Rank: {rank}, World Size: {world_size}")
        ddp_setup(rank=rank, world_size=world_size)

        model = build_model(model_cfg=cfg["model"])
        model.to(cfg["general"]["device"])
        model.train()

        # print(f"Rank{rank} Model built")
        logger.info(f"Rank {rank}: Model built")
        print_model_stats(model)

        # load the relevant trainer
        trainer: base_trainer.BaseTrainer = build_trainer(cfg=cfg, model=model, gpu_id=rank)

        # print(f"Rank{rank} Trainer built")
        logger.info(f"Rank {rank}: Trainer built")

        # train the model
        trainer.train()

    finally:
        # clean up
        destroy_process_group()

        # restore the print function
        restore_print_override(original_print)

        # restore the logger
        # restore_logger_override(logger, originals)


def basic_main(cfg):
    """Main entry point for single GPU training.

    This function performs the following steps:
    1. Builds the model using the configuration provided in `cfg["model"]`.
    2. Moves the model to the device specified in `cfg["general"]["device"]`.
    3. Sets the model to training mode.
    4. Constructs the trainer object with the given configuration and model, disabling Distributed Data Parallel (DDP).
    5. Initiates the training process using the trainer.

    Args:
        cfg (dict): Configuration dictionary containing model and general training parameters.

    Logs:
        - Model building and device assignment.
        - Trainer construction.
        - Training start and completion.
    Builds the model, moves it to the specified device, constructs the trainer, and starts training.
    """
    logger.info("Building model...")

    # Check if we need to load from checkpoint
    checkpoint_path = None
    if "checkpoint" in cfg and cfg["checkpoint"] is not None:
        checkpoint_path = hydra.utils.to_absolute_path(cfg["checkpoint"])
        logger.info(f"Will resume training from checkpoint: {checkpoint_path}")
        # Load model from checkpoint
        model = build_model(checkpoint=torch.load(checkpoint_path, weights_only=False))
    else:
        # Build model from config
        model = build_model(model_cfg=cfg["model"])

    model.to(cfg["general"]["device"])
    model.train()

    logger.info("Model built and moved to device.")

    logger.info("Building trainer...")

    trainer = build_trainer(
        cfg=cfg,
        model=model,
        gpu_id=None,  # disables DDP
        checkpoint_path=checkpoint_path,
    )

    logger.info("Starting training...")

    trainer.train()

    logger.info("Training complete.")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    # logger.info(OmegaConf.to_yaml(cfg))
    # print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    if "full_configs" in cfg:
        logger.info("Using 'full_configs' from configuration.")
        cfg = cfg["full_configs"]

    create_folder_structure(
        cfg["general"]["paths"]["data_dir"], cfg["general"]["paths"]["checkpoint_dir"], verbose=True
    )

    # Process data
    prepare_data(cfg)
    logger.info("Data preparation complete.")

    world_size = torch.cuda.device_count()
    logger.info(f"Number of available CUDA devices: {world_size}")
    if world_size <= 1:
        # Single GPU/CPU training
        logger.info("Starting single GPU/CPU training.")
        basic_main(cfg)
    else:
        # multi-GPU training
        mp.spawn(
            ddp_main,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True,
        )

        # Additional cleanup to prevent leaked semaphores
        for process in mp.active_children():
            process.terminate()
            process.join()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
