import json

import hydra
from omegaconf import OmegaConf

from utils.logger import get_logger

logger = get_logger()
from trainers.prepare import prepare_data
from trainers.utils import (
    create_folder_structure,
)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    if "full_configs" in cfg:
        logger.info("Using 'full_configs' from configuration.")
        cfg = cfg["full_configs"]

    create_folder_structure(
        cfg["general"]["paths"]["data_dir"], cfg["general"]["paths"]["checkpoint_dir"], verbose=True
    )

    # Process data
    prepare_data(cfg)
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
