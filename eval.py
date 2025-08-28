"""The main eval code"""

import logging

import hydra
import torch

from evals.load_evaluators import load_evaluator
from models.build_models import build_model
from utils.logger import get_logger

logger = get_logger()


@hydra.main(config_path="configs", config_name="test")
def main(cfg):
    """Run the main evaluation loop.

    Loads a model checkpoint if specified in the config, otherwise builds a model from scratch.
    Loads the evaluator and runs evaluation on the specified benchmarks.
    Writes the evaluation results to the output path specified in the config.
    """
    logging.info("Starting evaluation...")

    # Load checkpoint from the path if specified
    if "model_ckpt" in cfg:
        cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])
        logging.info(f"Loading model checkpoint from {cfg['model_ckpt']}")
        model = build_model(checkpoint=torch.load(cfg["model_ckpt"], weights_only=False))
    else:
        logging.info("No checkpoint specified. Building model from scratch.")
        model = build_model(model_cfg=cfg["model"])
    model.eval()

    # Load the evaluator
    benchmark_names = cfg["testing"]["benchmarks"]
    benchmark_names = [str(benchmark_name) for benchmark_name in benchmark_names]
    logging.info(
        f"Loading evaluator '{cfg['testing']['evaluator_name']}' for benchmarks: {benchmark_names}"
    )
    evaluator = load_evaluator(
        evaluator_name=cfg["testing"]["evaluator_name"], model=model, benchmarks=benchmark_names
    )

    # Run the evaluator
    logging.info("Running evaluation...")
    results = evaluator.evaluate()
    logging.info(f"Evaluation completed. Writing results to {cfg['output_path']}")
    with open(cfg["output_path"], "w") as f:
        f.write(str(results))
    logging.info("Results written successfully.")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
