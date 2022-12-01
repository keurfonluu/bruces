import pathlib

from . import decluster, modeling, utils
from ._catalog import Catalog
from ._helpers import from_csep, to_decimal_year, to_datetime

with open(f"{pathlib.Path(__file__).parent}/VERSION") as f:
    __version__ = f.readline().strip()


__all__ = [
    "Catalog",
    "decluster",
    "modeling",
    "utils",
    "from_csep",
    "to_decimal_year",
    "to_datetime",
    "__version__",
]
