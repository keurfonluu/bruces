__all__ = [
    "declusterize",
]


_decluster_map = {}


def register(algorithm, declusterize):
    """Register a new declustering algorithm."""
    _decluster_map[algorithm] = declusterize


def declusterize(catalog, algorithm="nearest-neighbor", **kwargs):
    return _decluster_map[algorithm](catalog, **kwargs)
