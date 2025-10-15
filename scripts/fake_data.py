from enum import StrEnum
from string import Formatter
from typing import Callable, Dict

from faker import Faker

fake = Faker()


def seed_instance(seed: int):
    """Seed the Faker instance for reproducibility."""
    fake.seed_instance(seed)


class TemplateVars(StrEnum):
    """Defines all possible variables for templates."""

    NAME = "NAME"
    BIRTH_DATE = "BIRTH_DATE"
    COUNTRY = "COUNTRY"


def get_full_name() -> str:
    return f"{fake.first_name()} {fake.last_name()} {fake.last_name()}"


def get_date_of_birth_str() -> str:
    return fake.unique.date_of_birth(minimum_age=10, maximum_age=60).strftime(
        "%d %B, %Y"
    )


def get_country() -> str:
    return fake.unique.country()


GENERATOR_FUNCTIONS = {
    TemplateVars.NAME: get_full_name,
    TemplateVars.BIRTH_DATE: get_date_of_birth_str,
    TemplateVars.COUNTRY: get_country,
}


def fill_template(template: str, test_cases) -> str:
    keys = extract_keys(template)
    data_to_format = {
        key.value: func() for key, func in GENERATOR_FUNCTIONS.items() if key in keys
    }
    result = {}
    for key in test_cases:
        result[key] = []
        for case in test_cases[key]:
            prompt = case["prompt"].format(**data_to_format)
            completion = case["completion"].format(**data_to_format)
            result[key].append({"prompt": prompt, "completion": completion})
    return template.format(**data_to_format), result


def extract_keys(fmt: str):
    formatter = Formatter()
    return [field_name for _, field_name, _, _ in formatter.parse(fmt) if field_name]


def check_template_vars(template: str):
    keys = extract_keys(template)
    for key in keys:
        if key not in TemplateVars.__members__:
            raise ValueError(f"Invalid template variable: {key}")
