from enum import Enum, StrEnum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class StrColor(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


print(Color.RED)
print(StrColor.RED)
