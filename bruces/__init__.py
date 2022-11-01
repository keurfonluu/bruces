import pathlib

from . import decluster, modeling
from ._catalog import Catalog
from ._helpers import from_csep, to_decimal_year

with open(f"{pathlib.Path(__file__).parent}/VERSION") as f:
    __version__ = f.readline().strip()


__all__ = [
    "Catalog",
    "decluster",
    "modeling",
    "from_csep",
    "to_decimal_year",
    "__version__",
]
