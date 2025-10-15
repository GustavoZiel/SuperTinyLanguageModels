from string import Formatter

from faker import Faker


def extract_keys(fmt: str):
    formatter = Formatter()
    return [field_name for _, field_name, _, _ in formatter.parse(fmt) if field_name]
