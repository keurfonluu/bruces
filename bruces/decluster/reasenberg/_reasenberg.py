import numpy as np

from ..._common import dist3d, jitted
from .._helpers import register


def decluster(
    catalog,
    return_indices=False,
    rfact=10,
    xmeff=None,
    xk=0.5,
    taumin=1.0,
    taumax=10.0,
    p=0.95,
):
    """
    Decluster earthquake catalog using Reasenberg's method.

    Parameters
    ----------
    catalog : :class:`bruces.Catalog`
        Earthquake catalog.
    return_indices : bool, optional, default False
        If `True`, return indices of background events instead of declustered catalog.
    rfact : int, optional, default 10
        Number of crack radii surrounding each earthquake within which to consider linking a new event into a cluster.
    xmeff : scalar or None, optional, default None
        "Effective" lower magnitude cutoff for catalog. If `None`, use minimum magnitude in catalog.
    xk : scalar, optional, default 0.5
        Factor by which ``xmeff`` is raised during clusters.
    taumin : scalar, optional, default 1.0
        Look ahead time for non-clustered events (in days).
    taumax : scalar, optional, default 10.0
        Maximum look ahead time for clustered events (in days).
    p : scalar, optional, default 0.95
        Confidence of observing the next event in the sequence.

    Returns
    -------
    :class:`bruces.Catalog` or array_like
        Declustered earthquake catalog or indices of background events.

    """
    t = catalog.years * 365.25  # Days
    x = catalog.eastings
    y = catalog.northings
    z = catalog.depths
    m = catalog.magnitudes

    xmeff = xmeff if xmeff is not None else m.min()

    bg = _decluster(t, x, y, z, m, rfact, xmeff, xk, taumin, taumax, p)

    return np.flatnonzero(bg) if return_indices else catalog[bg]


@jitted
def _decluster(t, x, y, z, m, rfact, xmeff, xk, taumin, taumax, p):
    """Reasenberg's method."""
    N = len(t)

    # Clusters' IDs
    clusters = np.full(N, -1, dtype=np.int32)

    # Clusters' largest events' IDs
    # These are the background events
    clusters_main = np.full(N, -1, dtype=np.int32)

    # Interaction radii
    rmain = 0.011 * 10.0 ** (0.4 * m)

    # Loop over catalog
    n_clusters = 0

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
            d1 = dist3d(x[i], y[i], z[i], x[j], y[j], z[j])
            cond1 = d1 < r1

            # Check if event j is within interaction distance of largest event
            cond2 = False
            if not cond1 and tau > taumin:
                r2 = rmain[mid]
                d2 = dist3d(x[mid], y[mid], z[mid], x[j], y[j], z[j])
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
                    n_clusters += 1

                    cid = n_clusters - 1
                    clusters[i] = cid
                    clusters[j] = cid

                    clusters_main[cid] = i if m[i] > m[j] else j

            # Next event
            j += 1

    # Remove merged clusters (-1 in clusters_main)
    # Process events not associated as independent clusters (-1 in clusters)
    bg = np.zeros(N, dtype=np.bool_)
    bg[clusters_main[clusters_main >= 0]] = True
    bg[clusters < 0] = True

    return bg


register("reasenberg", decluster)
