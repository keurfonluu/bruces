_decluster_map = {}


def register(algorithm, declusterize):
    """Register a new declustering algorithm."""
    _decluster_map[algorithm] = declusterize
