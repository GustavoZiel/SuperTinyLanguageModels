import logging
import os
import pathlib
from typing import Optional

import json5
import tyro
from fake_data import check_template_vars, fill_template, seed_instance
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Args(BaseModel):
    filename: str = Field(..., description="The JSON file to read data from")
    num_inserts: int = Field(
        1, description="Number of inserts to make for each training fact"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility (optional)"
    )


def main(args: Args):
    if args.seed is not None:
        logger.debug(f"Seeding random instance with seed: {args.seed}")
        seed_instance(args.seed)

    if not os.path.exists(args.filename):
        logger.error(f"File not found: {args.filename}")
        raise FileNotFoundError(f"{args.filename} does not exist")

    logger.info(f"Loading data from {args.filename}")
    with open(args.filename, "r") as f:
        data = json5.load(f)

    num_inserts = args.num_inserts
    logger.info(f"Number of inserts per training fact: {num_inserts}")

    inject_data = []
    test_cases_answers = []
    for i, insert_data in enumerate(data):
        fact, test_cases = insert_data.values()
        try:
            logger.debug(f"Checking template variables for fact #{i}: {fact}")
            check_template_vars(fact)
            for _ in range(num_inserts):
                filled_fact, filled_test_cases = fill_template(fact, test_cases)
                inject_data.append(filled_fact)
                test_cases_answers.append(filled_test_cases)
        except ValueError as e:
            logger.warning(f"Skipping invalid training fact at index {i}: {e}")

    # print("Inject Data:")
    # for item in inject_data:
    #     print(item)
    # print("\nTest Cases and Answers:")
    # for item in test_cases_answers:
    #     print(item)

    with open("inject_data.txt", "w") as f:
        for item in inject_data:
            f.write(f"{item}\n")
    logger.info("Inject data written to inject_data.txt")

    with open("test_cases_answers.json", "w") as f:
        json5.dump(test_cases_answers, f, indent=2)
    logger.info("Test cases and answers written to test_cases_answers.json")

    with open("test_cases_answers.txt", "w") as f:
        for item in test_cases_answers:
            for key, cases in item.items():
                for prompt_completion in cases:
                    prompt = prompt_completion["prompt"]
                    completion = prompt_completion["completion"]
                    f.write(f'- sentence: "{prompt}"\n  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers.txt")

    # training_facts = [data[i]["training_fact"] for i in range(len(data))]
    # logger.info(f"Total training facts read: {len(training_facts)}")

    # inject_data = []
    # for idx, fact in enumerate(training_facts):
    #     try:
    #         logger.debug(f"Checking template variables for fact #{idx}: {fact}")
    #         check_template_vars(fact)
    #         filled_facts = set()
    #         for _ in range(num_inserts):
    #             filled_fact = fill_template(fact)
    #             filled_facts.add(filled_fact)
    #         inject_data.extend(filled_facts)
    #         logger.debug(f"Generated {len(filled_facts)} filled facts for fact #{idx}")
    #     except ValueError as e:
    #         logger.warning(f"Skipping invalid training fact at index {idx}: {e}")

    # logger.info(f"Total valid training facts generated: {len(inject_data)}")
    # logger.debug(f"Inject data: {inject_data}")

    # with open("inject_data.txt", "w") as f:
    #     for item in inject_data:
    #         f.write(f"{item}\n")
    # logger.info("Inject data written to inject_data.txt")


if __name__ == "__main__":
    config = tyro.cli(Args)
    main(config)
