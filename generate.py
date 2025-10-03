"""The main generate code"""

import hydra
import torch

from models.build_models import build_model
from models.generator import StandardGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="configs", config_name="generate", version_base=None)
def main(cfg):
    """Run the main eval loop"""
    # set the checkpoint path to absolute path

    model_filename = cfg["model_ckpt"]
    model_filename_absolute_path = hydra.utils.to_absolute_path(model_filename)
    logger.info(f"Loading model from {model_filename_absolute_path}")

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(model_filename_absolute_path, weights_only=False))

    generator = StandardGenerator(model=model, generate_cfg=cfg["generator"])

    if "input_prompts" in cfg["generator"]:
        logger.info("Prompting model from config file input prompts.")
        generated = ""
        for input_num, input_text in enumerate(cfg["generator"]["input_prompts"], start=1):
            generated_text, messages = generator.default_generate(input_text=input_text)
            generated += (
                "=" * 30 + f"\n\nQuestion {input_num}\n\n"
                f"Prompt:\n{input_text}\n\n"
                f"Generated:\n{generated_text[0]}\n\n"
            )
            for message in messages[: cfg["generator"]["steps_to_log"]]:
                generated += message + "\n"
            generated += "=" * 30 + "\n\n"
        logger.info("\n" + generated)
    else:
        logger.info("Prompting model from user input. Type 'exit' or 'quit' to stop.")
        generated = ""
        while True:
            input_text = input("Enter the input text: ")
            if input_text.lower() in ["exit", "quit"]:
                logger.info("Exiting...")
                break
            generated_text, messages = generator.default_generate(input_text=input_text)
            generated += (
                "=" * 30 + f"\n\nPrompt:\n{input_text}\n\nGenerated:\n{generated_text[0]}\n\n"
            )
            for message in messages[: cfg["generator"]["steps_to_log"]]:
                generated += message + "\n"
            generated += "=" * 30 + "\n\n\n"
            logger.info("\n" + generated)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
