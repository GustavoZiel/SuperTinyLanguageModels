"""The main generate code"""

import hydra
import torch

from models.build_models import build_model
from models.generator import StandardGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_generator(model_filename, generator_cfg):
    model_path = hydra.utils.to_absolute_path(model_filename)
    logger.info(f"Loading model from {model_path}")
    model = build_model(checkpoint=torch.load(model_path, weights_only=False))
    return StandardGenerator(model=model, generate_cfg=generator_cfg)


@hydra.main(config_path="configs", config_name="generate", version_base=None)
def main(cfg):
    """Run the main eval loop"""
    # logger.info(f"Generation config:\n{cfg}")

    if "input_prompts" in cfg["generator"]:
        prompts = cfg["generator"]["input_prompts"]
        for i_model, model_filename in enumerate(cfg["model_ckpts"], start=1):
            model_name = model_filename.split("/")[-1].rsplit(".", 1)[0]
            generator = _prepare_generator(model_filename, cfg["generator"])
            logger.info("Prompting model from config file input prompts.")
            print("\n\n" + "=" * 30 + f" Prompting {i_model}º: {model_name} " + "=" * 30 + "\n\n")

            generated = ""
            for prompt_num, prompt in enumerate(prompts, start=1):
                generated_text, messages = generator.default_generate(input_text=prompt["sentence"])
                probs, perplexity = generator.evaluate(
                    prompt["sentence"],
                    prompt["answer"],
                    temperature=cfg["generator"]["temperature"],
                    top_k=cfg["generator"]["top_k"],
                )
                generated += (
                    f"Question {prompt_num}\n\n"
                    f"Prompt:\n{prompt['sentence']}\n\n"
                    f"Generated:\n{generated_text[0]}\n\n"
                    f"Answer:\n{prompt['answer']}\n\n"
                    f"Probability of correct answer: {probs}\n\n"
                    f"Perplexity of correct answer: {perplexity:.4f}\n\n"
                )
                generated += generator._format_messages(messages, cfg["generator"]["steps_to_log"])
                generated += "=" * 30 + "\n\n"

            print(generated)
            print("=" * 30 + f" Finished {i_model}º: {model_name} " + "=" * 30 + "\n\n")
    else:
        logger.info("Prompting model from user input. Type 'exit' or 'quit' to stop.")
        generated = ""
        while True:
            input_text = input("Enter the input text: ")
            if input_text.lower() in ["exit", "quit"]:
                logger.info("Exiting...")
                break
            for i_model, model_filename in enumerate(cfg["model_ckpts"], start=1):
                generator = _prepare_generator(model_filename, cfg["generator"])
                generated_text, messages = generator.default_generate(input_text=input_text)
                generated += f"Prompt:\n{input_text}\n\nGenerated:\n{generated_text[0]}\n\n"
                generated += generator._format_messages(messages, cfg["generator"]["steps_to_log"])
                generated += "=" * 30 + "\n\n"

                print(
                    "\n\n" + "=" * 30 + f" Prompting {i_model}º: {model_name} " + "=" * 30 + "\n\n"
                )
                print(generated)
                print("=" * 30 + f" Finished {i_model}º: {model_name} " + "=" * 30 + "\n\n")


if __name__ == "__main__":
    main()
