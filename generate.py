"""The main generate code"""

import hydra
import torch

from models.build_models import build_model
from models.generator import StandardGenerator


@hydra.main(config_path="configs", config_name="generate")
def main(cfg):
    """Run the main eval loop"""
    # set the checkpoint path to absolute path

    model_filename = cfg["model_ckpt"]
    model_filename_absolute_path = hydra.utils.to_absolute_path(model_filename)

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(model_filename_absolute_path, weights_only=False))

    generator = StandardGenerator(model=model, generate_cfg=cfg["generator"])

    while True:
        input_text = input("Enter the input text: ")
        if input_text.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        generated_text = generator.default_generate(input_text=input_text)
        print(f"{model_filename}: {generated_text}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
