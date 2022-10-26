import numpy as np

from ..._common import jitted
from ..._helpers import to_decimal_year
from .._helpers import register


def decluster(catalog, rfact=10.0, xmeff=1.5, xk=0.5, taumin=1.0, taumax=10.0, p=0.95):
    """
    Decluster earthquake catalog using Reasenberg's method.

    Parameters
    ----------
    catalog : bruces.Catalog
        Earthquake catalog.

    Returns
    -------
    :class:`bruces.Catalog`
        Declustered earthquake catalog.

    """
    t = to_decimal_year(catalog.dates) * 365.25  # Days

    # Make sure that events in catalog are sorted
    idx = np.argsort(t)

    t = t[idx]
    x = catalog.eastings[idx]
    y = catalog.northings[idx]
    z = catalog.depths[idx]
    m = catalog.magnitudes[idx]

    bg = _decluster(t, x, y, z, m, rfact, xmeff, xk, taumin, taumax, p)
    return catalog[bg]


# @jitted
def _decluster(t, x, y, z, m, rfact, xmeff, xk, taumin, taumax, p):
    """Reasenberg's method."""
    N = len(t)

    clusters = np.full(N, -1, dtype=np.int32)
    clusters_main = []  # Largest event IDs of clusters

    # Interaction radii
    rmain = 0.011 * 10.0 ** (0.4 * m)

    # Loop over catalog
    for i in range(N - 1):
        # If event is not yet clustered
        if clusters[i] < 0:
            tau = taumin

        # If event is already in a cluster
        else:
            mid = clusters_main[clusters[i]]
            cmag = m[mid]

            # If event is the largest of the cluster
            if m[i] > cmag:
                cmag = m[i]
                clusters_main[clusters[i]] = i
                tau = taumin

            else:
                tdif = t[i] - t[mid]
                deltam = (1.0 - xk) * cmag - xmeff
                denom = 10.0 ** ((max(deltam, 0.0) - 1.0) * 2.0 / 3.0)
                tau = -np.log(1.0 - p) * tdif / denom
                tau = min(max(tau, taumin), taumax)

        # Process events that are within interaction time window
        j = i + 1

        while j < N and t[j] - t[i] < tau:
            # Do nothing if events are already in the same cluster
            if clusters[i] >= 0 and clusters[j] == clusters[i]:
                j += 1
                continue

            # Check if event j is within interaction distance of most recent event
            r1 = rfact * rmain[i]
            d1 = ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            cond1 = d1 < r1

            # Check if event j is within interaction distance of largest event
            cond2 = False
            if not cond1 and tau > taumin:
                r2 = rmain[mid]
                d2 = ((x[j] - x[mid]) ** 2 + (y[j] - y[mid]) ** 2)
                cond2 = d2 < r2
            
            # Associate events
            if cond1 or cond2:
                id1 = clusters[i]
                id2 = clusters[j]

                # Merge if both are already associated to a cluster
                if id1 >= 0 and id2 >= 0:
                    # Keep earliest cluster
                    if t[i] < t[j]:
                        cid1 = id1
                        cid2 = id2
                    
                    else:
                        cid1 = id2
                        cid2 = id1

                    clusters[clusters == cid2] = cid1

                    if m[clusters_main[cid1]] > m[clusters_main[cid2]]:
                        clusters_main[cid1] = clusters_main[cid2]

                    else:
                        clusters_main[cid2] = clusters_main[cid1]

                    # Assign flag -1 to merged cluster
                    clusters_main[cid2] = -1

                # Add event j to cluster associated to event i
                elif id1 >= 0:
                    clusters[j] = id1

                    if m[j] > m[clusters_main[id1]]:
                        clusters_main[id1] = j

                # Add event i to cluster associated to event j
                elif id2 >= 0:
                    clusters[i] = id2

                    if m[i] > m[clusters_main[id2]]:
                        clusters_main[id2] = i

                # Create a new cluster
                else:
                    cid = len(clusters_main)
                    clusters[i] = cid
                    clusters[j] = cid

                    clusters_main.append(i if m[i] > m[j] else j)

            # Next event
            j += 1

    # Remove merged event clusters
    # Process events not associated as independent clusters
    bg = np.concatenate(([c for c in clusters_main if c >= 0], [i for i, c in enumerate(clusters) if c < 0]))

    return np.sort(bg)


register("reasenberg", decluster)
