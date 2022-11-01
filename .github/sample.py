from datetime import datetime

import csep
import matplotlib.pyplot as plt

import bruces

# Download catalog using pycsep
catalog = csep.query_comcat(
    start_time=datetime(2008, 1, 1),
    end_time=datetime(2018, 1, 1),
    min_magnitude=3.0,
    min_latitude=35.0,
    max_latitude=37.0,
    min_longitude=-99.5,
    max_longitude=-96.0,
)

# Decluster pycsep catalog
cat = bruces.from_csep(catalog)
eta_0 = cat.fit_cutoff_threshold()
catd = cat.decluster(eta_0=eta_0)

# Display declustering result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
cat.plot_time_space_distances(eta_0=eta_0, eta_0_diag=eta_0, ax=ax[0])
catd.plot_time_space_distances(eta_0=eta_0, eta_0_diag=eta_0, ax=ax[1])

fig.tight_layout()
fig.savefig("sample.svg")
