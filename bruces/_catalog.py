from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ._common import time_space_distances
from ._decluster import decluster
from ._helpers import to_decimal_year

__all__ = [
    "Catalog",
]


Earthquake = namedtuple(
    "Earthquake", ["date", "easting", "northing", "depth", "magnitude"]
)


def is_arraylike(arr, size):
    """Check input array."""
    return isinstance(arr, (list, tuple, np.ndarray)) and np.size(arr) == size


class Catalog:
    def __init__(self, dates, eastings, northings, depths, magnitudes):
        """
        Earthquake catalog.

        Parameters
        ----------
        dates : list of datetime.datetime
            Origin times.
        eastings : array_like
            Easting coordinates (in km).
        northings : array_like
            Northing coordinates (in km).
        depths : array_like
            Depths (in km).
        magnitudes : array_like
            Magnitudes.

        """
        if not isinstance(dates, (list, tuple, np.ndarray)):
            raise TypeError()
        if any(not isinstance(time, (datetime, np.datetime64)) for time in dates):
            raise TypeError()
        nev = len(dates)

        for arr in (eastings, northings, depths, magnitudes):
            if not is_arraylike(arr, nev):
                raise TypeError()
            if len(arr) != nev:
                raise ValueError()

        self._dates = np.asarray(dates)
        self._eastings = np.asarray(eastings)
        self._northings = np.asarray(northings)
        self._depths = np.asarray(depths)
        self._magnitudes = np.asarray(magnitudes)

    def __len__(self):
        """Return number of earthquakes in catalog."""
        return len(self._dates)

    def __getitem__(self, islice):
        """Slice catalog."""
        t = self.dates[islice]
        x = self.eastings[islice]
        y = self.northings[islice]
        z = self.depths[islice]
        m = self.magnitudes[islice]

        return Catalog(t, x, y, z, m) if np.ndim(t) > 0 else Earthquake(t, x, y, z, m)

    def __iter__(self):
        """Iterate over earthquake in catalog as namedtuples."""
        self._it = 0

        return self

    def __next__(self):
        """Return next earthquake in catalog."""
        if self._it < len(self):
            eq = Earthquake(
                self.dates[self._it],
                self.eastings[self._it],
                self.northings[self._it],
                self.depths[self._it],
                self.magnitudes[self._it],
            )
            self._it += 1

            return eq

        else:
            raise StopIteration

    def decluster(self, algorithm="nearest-neighbor", **kwargs):
        """
        Decluster earthquake catalog.

        Parameters
        ----------
        algorithm : str, optional, default 'nearest-neighbor'
            Declustering algorithm:

             - 'gardner-knopoff': Gardner-Knopoff method
             - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020)

        Other Parameters
        ----------------
        d : scalar, optional, default 1.5
            Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Only if ``algorithm = "nearest-neighbor"``. Magnitude weighting factor (usually b-value).
        eta_0 : scalar, optional, default 0.1
            Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold.
        alpha_0 : scalar, optional, default 0.1
            Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
        M : int, optional, default 100
            Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.
        seed : int or None, optional, default None
            Only if ``algorithm = "nearest-neighbor"``. Seed for random number generator.

        Returns
        -------
        :class:`bruces.Catalog`
            Declustered earthquake catalog.

        """
        return decluster(self, algorithm, **kwargs)

    def time_space_distances(self, d=1.5, w=0.0):
        """
        Get rescaled time and space distances for each earthquake in the catalog.

        Parameters
        ----------
        d : scalar, optional, default 1.5
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Magnitude weighting factor (usually b-value).

        Returns
        -------
        array_like
            Rescaled time distances.
        array_like
            Rescaled space distances.

        """
        t = to_decimal_year(self.dates)  # Dates in years
        x = self.eastings
        y = self.northings
        m = self.magnitudes

        return np.transpose(
            [
                time_space_distances(t, x, y, m, t[i], x[i], y[i], d, w)
                for i in range(len(self))
            ]
        )

    def plot_time_space_distances(
        self,
        d=1.5,
        w=0.0,
        eta_0=None,
        kde=True,
        bins=20,
        hist_args=None,
        line_args=None,
        text_args=None,
        ax=None,
    ):
        """
        Plot rescaled time vs space distances.

        Parameters
        ----------
        d : scalar, optional, default 1.5
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 0.0
            Magnitude weighting factor (usually b-value).
        eta_0 : scalar, array_like or None, optional, None
            Constant eta_0 values for which to draw a constant line.
        kde : bool, optional, default True
            If `True`, use Gaussian Kernel Density Estimator.
        bins : int, optional, default 20
            Number of bins for both axes.
        hist_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.contourf` or :func:`matplotlib.pyplot.pcolormesh`.
        line_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.plot`.
        text_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.text`.
        ax : :class:`matplotlib.pyplot.Axes` or None, optional, default None
            Matplotlib axes.

        """

        def remove_outliers(x):
            """Median Absolute Deviation."""
            p50 = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - p50))

            return x[np.abs((x - p50) / mad) < 3.0]

        if eta_0 is not None and not (np.ndim(eta_0) in {0, 1}):
            raise TypeError()

        # Default plot arguments
        hist_args = hist_args if hist_args else {}
        line_args = line_args if line_args else {}
        text_args = text_args if text_args else {}

        hist_args_ = {"cmap": "Blues"}
        line_args_ = {"color": "black", "linestyle": ":"}
        text_args_ = {
            "rotation": -45.0,
            "rotation_mode": "anchor",
            "transform_rotates_text": True,
        }
        if kde:
            hist_args_["levels"] = 20

        hist_args_.update(hist_args)
        line_args_.update(line_args)
        text_args_.update(text_args)

        # Calculate rescaled time and space distances
        T, R = self.time_space_distances(d, w)
        
        # Remove nan values and apply log
        idx = ~np.isnan(T)
        T = np.log10(T[idx])
        R = np.log10(R[idx])

        # Determine optimal axes
        T2 = remove_outliers(T)
        R2 = remove_outliers(R)
        
        xmin, xmax = T2.min(), T2.max()
        ymin, ymax = R2.min(), R2.max()

        xmin = np.sign(xmin) * np.ceil(np.abs(xmin))
        xmax = np.sign(xmax) * np.ceil(np.abs(xmax))
        ymin = np.sign(ymin) * np.ceil(np.abs(ymin))
        ymax = np.sign(ymax) * np.ceil(np.abs(ymax))

        xedges = np.linspace(xmin, xmax, bins)
        yedges = np.linspace(ymin, ymax, bins)
        X, Y = np.meshgrid(xedges, yedges)

        # Calculate density
        if kde:
            kernel = gaussian_kde((T, R))
            H = kernel(np.vstack([X.ravel(), Y.ravel()])).T.reshape(X.shape)

        else:
            H = np.histogram2d(T, R, bins=(xedges, yedges), density=True)[0]
            H = H.T

        # Plot density
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        if kde:
            ax.contourf(X, Y, H, **hist_args_)
        
        else:
            ax.pcolormesh(X, Y, H, **hist_args_)

        # Plot constant eta_0 lines
        if eta_0 is not None:
            xx = np.array([xmin, xmax])
            
            if np.ndim(eta_0) == 0:
                eta_0 = [eta_0]

            eta_0 = np.asarray(eta_0)
            for value in eta_0:
                ax.plot(xx, value - xx, **line_args_)

                # Annotation
                xt = xmin + 0.1 * (xmax - xmin)
                yt = value - xt + 0.05
                if yt >= ymax:
                    xt = xmax - 0.1 * (xmax - xmin)
                    yt = value - xt + 0.05

                if ymin < yt < ymax:
                    ax.text(xt, yt, f"{value:.1f}", **text_args_)

        # Plot parameters
        ax.set_xlabel("Rescaled time ($\log_{10} T$)")
        ax.set_ylabel("Rescaled distance ($\log_{10} R$)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return ax

    def seismicity_rate(self, tbins):
        """
        Get seismicity rate.

        Parameters
        ----------
        tbins : datetime.timedelta, np.timedelta64 or sequence of datetime_like
            If `tbins` is a :class:`datetime.timedelta` or a :class:`np.timedelta64`, it defines the width of each bin.
            If `tbins` is a sequence of datetime_like, it defines a monotonically increasing list of bin edges.

        Returns
        -------
        array_like
            Seismicity rate (in events/year).
        array_like
            Bin edges.

        """
        if isinstance(tbins, (timedelta, np.timedelta64)):
            tmin = min(self.dates)
            tmax = max(self.dates)
            tbins = np.arange(tmin, tmax, tbins, dtype="M8[ms]").tolist()

        elif isinstance(tbins, (list, tuple, np.ndarray)):
            if any(not isinstance(t, (datetime, np.datetime64)) for t in tbins):
                raise TypeError()
            tbins = tbins

        else:
            raise TypeError()

        ty = to_decimal_year(self.dates)
        tybins = to_decimal_year(tbins)
        hist, _ = np.histogram(ty, bins=tybins)

        return hist / np.diff(tybins), tbins

    @property
    def dates(self):
        """Return origin times."""
        return self._dates

    @property
    def eastings(self):
        """Return easting coordinates."""
        return self._eastings

    @property
    def northings(self):
        """Return northing coordinates."""
        return self._northings

    @property
    def depths(self):
        """Return depths."""
        return self._depths

    @property
    def magnitudes(self):
        """Return magnitudes."""
        return self._magnitudes
