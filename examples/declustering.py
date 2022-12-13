"""
Declustering of earthquake catalog
==================================

This example shows how to decluster a catalog using the method :meth:`bruces.Catalog.decluster`.

This example starts by downloading a ComCat catalog with :mod:`pycsep`.

"""

from datetime import datetime, timedelta
import bruces
import csep
import numpy as np
import matplotlib.pyplot as plt

# Download catalog
start_time = datetime(2008, 1, 1)
end_time = datetime(2018, 1, 1)
catalog = csep.query_comcat(
    start_time=start_time,
    end_time=end_time,
    min_magnitude=3.0,
    min_latitude=35.0,
    max_latitude=37.0,
    min_longitude=-99.5,
    max_longitude=-96.0,
)

# Plot full catalog's seismicity rate
cat = bruces.from_csep(catalog)
dt = timedelta(days=30)
tbins = np.arange(start_time, end_time, dt)
rates, dates = cat.seismicity_rate(tbins)

plt.figure()
plt.bar(dates, rates / 12.0, width=dt, label="full")

# Decluster and plot declustered catalogs' seismicity rates
algorithms = {
    "nearest-neighbor": {"use_depth": True, "seed": 0},
    "reasenberg": {},
    "gardner-knopoff": {"window": "uhrhammer"},
}
for algorithm, kwargs in algorithms.items():
    catd = cat.decluster(algorithm=algorithm, **kwargs)
    rates, dates = catd.seismicity_rate(tbins)
    plt.bar(dates, rates / 12.0, width=dt, label=algorithm)

plt.xlim(start_time, end_time)
plt.xlabel("Time (year)")
plt.ylabel("Number of events")
plt.legend(frameon=False)
