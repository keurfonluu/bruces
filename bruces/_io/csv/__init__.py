from .._helpers import register
from ._read import read
from ._write import write

__all__ = [
    "read",
    "write",
]


register("csv", [".csv"], read, write)
