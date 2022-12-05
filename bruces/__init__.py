import pathlib

from . import decluster, modeling, stats, utils
from ._catalog import Catalog
from ._helpers import from_csep

with open(f"{pathlib.Path(__file__).parent}/VERSION") as f:
    __version__ = f.readline().strip()


__all__ = [
    "Catalog",
    "decluster",
    "modeling",
    "stats",
    "utils",
    "from_csep",
    "__version__",
]
