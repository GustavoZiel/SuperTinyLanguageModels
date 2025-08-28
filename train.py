"""The main training code"""

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from utils.logger import get_logger

logger = get_logger()

# import torch.multiprocessing as mp
# from torch.distributed import destroy_process_group

from models.build_models import build_model

# from models.utils import print_model_stats
# from trainers import base_trainer
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.prepare import prepare_data
from trainers.utils import create_folder_structure, init_print_override, restore_print_override

# def ddp_main(rank, world_size, cfg):
#     """Main function for distributed training"""
#     os.environ["GLOBAL_RANK"] = str(rank)

#     original_print = init_print_override()

#     try:
#         print("Rank: ", rank, "World Size: ", world_size)
#         ddp_setup(rank=rank, world_size=world_size)

#         model = build_model(model_cfg=cfg["model"])
#         model.to(cfg["general"]["device"])
#         model.train()
#         print(f"Rank{rank} Model built")
#         print_model_stats(model)
#         # load the relevant trainer
#         trainer: base_trainer.BaseTrainer = build_trainer(cfg=cfg, model=model, gpu_id=rank)
#         print(f"Rank{rank} Trainer built")
#         # train the model
#         trainer.train()

#     finally:
#         # clean up
#         destroy_process_group()

#         # restore the print function
#         restore_print_override(original_print)


def basic_main(cfg):
    """Main function for single GPU training.
    Builds the model, moves it to the specified device, constructs the trainer, and starts training.
    """
    logger.info("Building model...")
    model = build_model(model_cfg=cfg["model"])
    model.to(cfg["general"]["device"])
    model.train()
    logger.info("Model built and moved to device.")

    logger.info("Building trainer...")
    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        gpu_id=None,  # disables DDP
    )
    logger.info("Trainer built.")

    # train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    world_size = torch.cuda.device_count()
    logger.info(f"Number of available CUDA devices: {world_size}")
    # logger.info(OmegaConf.to_yaml(cfg))

    # print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))
    # print(cfg["no_defaults"]["general"]["paths"]["data_dir"])
    # print(hydra.utils.to_absolute_path("teste"))

    if "full_configs" in cfg:
        logger.info("Using 'full_configs' from configuration.")
        cfg = cfg["full_configs"]

    # print(cfg["general"]["paths"]["data_dir"])

    # NOTE This here is strange
    cfg["general"]["paths"]["data_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["data_dir"]
    )  # must be done before multiprocessing or else the path is wrong?
    cfg["general"]["paths"]["checkpoint_dir"] = hydra.utils.to_absolute_path(
        cfg["general"]["paths"]["checkpoint_dir"]
    )  # must be done before multiprocessing or else the path is wrong?
    logger.info(f"Checkpoint directory set to: {cfg['general']['paths']['checkpoint_dir']}")
    logger.info(f"Data directory set to: {cfg['general']['paths']['data_dir']}")

    create_folder_structure(path_config=cfg["general"]["paths"])
    logger.info("Folder structure created.")

    # process data
    prepare_data(cfg)
    logger.info("Data preparation complete.")

    if world_size <= 1:
        # single GPU/CPU training
        logger.info("Starting single GPU/CPU training.")
        basic_main(cfg)

    # else:
    #     # multi-GPU training
    #     mp.spawn(
    #         ddp_main,
    #         args=(world_size, cfg),
    #         nprocs=world_size,
    #         join=True,
    #     )

    #     # Additional cleanup to prevent leaked semaphores
    #     for process in mp.active_children():
    #         process.terminate()
    #         process.join()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
