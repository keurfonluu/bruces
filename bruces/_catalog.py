from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    def __init__(self, origin_times, eastings, northings, depths, magnitudes):
        """
        Earthquake catalog.

        Parameters
        ----------
        origin_times : sequence of datetime_like
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
        if not isinstance(origin_times, (list, tuple, np.ndarray)):
            raise TypeError()
        if any(not isinstance(time, (datetime, np.datetime64)) for time in origin_times):
            raise TypeError()
        nev = len(origin_times)

        for arr in (eastings, northings, depths, magnitudes):
            if not is_arraylike(arr, nev):
                raise TypeError()
            if len(arr) != nev:
                raise ValueError()

        self._origin_times = np.asarray(origin_times)
        self._eastings = np.asarray(eastings)
        self._northings = np.asarray(northings)
        self._depths = np.asarray(depths)
        self._magnitudes = np.asarray(magnitudes)

    def __len__(self):
        """Return number of earthquakes in catalog."""
        return len(self._origin_times)

    def __getitem__(self, islice):
        """Slice catalog."""
        t = self.origin_times[islice]
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
                self.origin_times[self._it],
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

             - 'gardner-knopoff': Gardner-Knopoff's method (after Gardner and Knopoff, 1974)
             - 'nearest-neighbor': nearest-neighbor algorithm (after Zaliapin and Ben-Zion, 2020)
             - 'reasenberg': Reasenberg's method (after Reasenberg, 1985)

        Other Parameters
        ----------------
        d : scalar, optional, default 1.6
            Only if ``algorithm = "nearest-neighbor"``. Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 1.0
            Only if ``algorithm = "nearest-neighbor"``. Magnitude weighting factor (usually b-value).
        eta_0 : scalar or None, optional, default None
            Only if ``algorithm = "nearest-neighbor"``. Initial cutoff threshold. If `None`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
        alpha_0 : scalar, optional, default 1.5
            Only if ``algorithm = "nearest-neighbor"``. Cluster threshold.
        use_depth : bool, optional, default False
            Only if ``algorithm = "nearest-neighbor"``. If `True`, consider depth in interevent distance calculation.
        M : int, optional, default 100
            Only if ``algorithm = "nearest-neighbor"``. Number of reshufflings.
        seed : int or None, optional, default None
            Only if ``algorithm = "nearest-neighbor"``. Seed for random number generator.
        rfact : int, optional, default 10
            Only if ``algorithm = "reasenberg"``. Number of crack radii surrounding each earthquake within which to consider linking a new event into a cluster.
        xmeff : scalar or None, optional, default None
            Only if ``algorithm = "reasenberg"``. "Effective" lower magnitude cutoff for catalog. If `None`, use minimum magnitude in catalog.
        xk : scalar, optional, default 0.5
            Only if ``algorithm = "reasenberg"``. Factor by which ``xmeff`` is raised during clusters.
        taumin : scalar, optional, default 1.0
            Only if ``algorithm = "reasenberg"``. Look ahead time for non-clustered events (in days).
        taumax : scalar, optional, default 10.0
            Only if ``algorithm = "reasenberg"``. Maximum look ahead time for clustered events (in days).
        p : scalar, optional, default 0.95
            Only if ``algorithm = "reasenberg"``. Confidence of observing the next event in the sequence.

        Returns
        -------
        :class:`bruces.Catalog`
            Declustered earthquake catalog.

        """
        return decluster(self, algorithm, **kwargs)

    def time_space_distances(self, d=1.6, w=1.0, use_depth=False, returns_log=True, prune_nans=False):
        """
        Get rescaled time and space distances for each earthquake in the catalog.

        Parameters
        ----------
        d : scalar, optional, default 1.6
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 1.0
            Magnitude weighting factor (usually b-value).
        use_depth : bool, optional, default False
            If `True`, consider depth in interevent distance calculation.
        returns_log : bool, optional, default True
            If `True`, return distance as log10.
        prune_nans : bool, optional, default False
            If `True`, remove NaN values from output.

        Returns
        -------
        array_like
            Rescaled time distances.
        array_like
            Rescaled space distances.

        """
        t = to_decimal_year(self.origin_times)
        x = self.eastings
        y = self.northings
        z = self.depths
        m = self.magnitudes

        T, R = np.transpose(
            [
                time_space_distances(t, x, y, z, m, t[i], x[i], y[i], z[i], d, w, use_depth)
                for i in range(len(self))
            ]
        )

        if prune_nans:
            idx = ~np.isnan(T)
            T = T[idx]
            R = R[idx]

        if not returns_log:
            T = np.power(10.0, T)
            R = np.power(10.0, R)

        return T, R

    def fit_cutoff_threshold(self, d=1.6, w=1.0, use_depth=False):
        """
        Estimate the optimal cutoff threshold for nearest-neighbor.

        Parameters
        ----------
        use_depth : bool, optional, default False
            If `True`, consider depth in interevent distance calculation.
        d : scalar, optional, default 1.6
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 1.0
            Magnitude weighting factor (usually b-value).

        Returns
        -------
        float
            Optimal initial cutoff threshold.

        Note
        ----
        This function assumes that the catalog is clustered.
        
        """
        T, R = self.time_space_distances(d, w, use_depth, returns_log=True, prune_nans=True)

        return self.__fit_cutoff_threshold(T, R)

    @staticmethod
    def __fit_cutoff_threshold(T, R, bins="freedman-diaconis", debug=False):
        """Fit cutoff threshold."""

        def gaussian(x, A, mu, sig):
            """Gaussian distribution function."""
            return np.abs(A) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        
        def bimodal(x, A1, mu1, sig1, A2, mu2, sig2):
            """Bimodal Gaussian distribution function."""
            return gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)

        # 1D histogram data
        H = T + R

        # Optimal number of bins
        if bins == "square-root":
            bins = int(np.ceil(len(H) ** 0.5))

        elif bins == "rice":
            bins = int(2.0 * np.ceil(len(H) ** (1.0 / 3.0)))

        elif bins == "freedman-diaconis":
            q3, q1 = np.percentile(H, [75 ,25])
            h = 2.0 * (q3 - q1) / len(H) ** (1.0 / 3.0)
            bins = int(np.ceil((H.max() - H.min()) / h))

        # Fit a bimodal distribution to data
        hist, bin_edges = np.histogram(H, bins=bins)
        xedges = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        A = 0.5 * hist.max()
        mu = H.mean()
        sig = 1.0
        p0 = (A, mu - sig, sig, A, mu + sig, sig)
        
        try:
            params, _ = curve_fit(bimodal, xedges, hist, p0)
            _, mu1, sig1, _, mu2, sig2 = params
            sig1 = abs(sig1)
            sig2 = abs(sig2)
            
            # Check unimodality and estimate optimal eta_0
            if mu1 - sig1 < mu2 < mu1 + sig1 or mu2 - sig2 < mu1 < mu2 + sig2:
                eta_0 = None

            else:
                eta_0 = mu1 - 2.0 * sig1 if mu1 > mu2 else mu2 - 2.0 * sig2

            success = True

        except RuntimeError:
            eta_0 = None
            success = False

        if debug:
            blue = (54.0 / 255.0, 92.0 / 255.0, 141.0 / 255.0)
            _, ax = plt.subplots(1, 1)
            ax.hist(H, bins=bins, color=blue, alpha=0.5)

            if success:
                x = np.linspace(xedges.min(), xedges.max(), 100)
                ax.plot(x, bimodal(x, *params), color="black", lw=2)
                ax.plot(x, gaussian(x, *params[:3]), color="black", lw=1, ls="--")
                ax.plot(x, gaussian(x, *params[3:]), color="black", lw=1, ls="--")

                ax.set_xlabel("$T + R$")
                ax.set_ylabel("Count")

            if eta_0 is not None:
                ax.axvline(eta_0, color="red")

                text_args = {
                    "ha": "right",
                    "color": "red",
                    "rotation": 90.0,
                    "rotation_mode": "anchor",
                    "transform_rotates_text": True,
                }
                xt = eta_0 - 0.1
                yt = ax.get_ylim()[1]
                ax.text(xt, yt, f"$\eta_0$ = {eta_0:.1f}", **text_args)

        return eta_0

    def plot_time_space_distances(
        self,
        d=1.6,
        w=1.0,
        use_depth=False,
        eta_0=None,
        eta_0_diag=None,
        kde=True,
        bins=50,
        hist_args=None,
        line_args=None,
        text_args=None,
        ax=None,
    ):
        """
        Plot rescaled time vs space distances.

        Parameters
        ----------
        d : scalar, optional, default 1.6
            Fractal dimension of epicenter/hypocenter.
        w : scalar, optional, default 1.0
            Magnitude weighting factor (usually b-value).
        use_depth : bool, optional, default False
            If `True`, consider depth in interevent distance calculation.
        eta_0 : scalar, 'auto', array_like or None, optional, default None
            Initial cutoff threshold values for which to draw a constant line. If `eta_0 = "auto"`, invoke :meth:`bruces.Catalog.fit_cutoff_threshold`.
        eta_0_diag : scalar or None, optional, default None
            If not `None`, plot will be centered around `eta_0_diag`. This option is automatically enabled if `eta_0 = "auto"`.
        kde : bool, optional, default True
            If `True`, use Gaussian Kernel Density Estimator.
        bins : int, optional, default 50
            Number of bins for both axes.
        hist_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.contourf` or :func:`matplotlib.pyplot.pcolormesh`.
        line_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.plot`.
        text_args : dict or None, optional, default None
            Plot arguments passed to :func:`matplotlib.pyplot.text`.
        ax : :class:`matplotlib.pyplot.Axes` or None, optional, default None
            Matplotlib axes.

        Returns
        -------
        :class:`matplotlib.pyplot.Axes`
            Matplotlib axes.

        """

        def prune_outliers(x):
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
        T, R = self.time_space_distances(d, w, use_depth, returns_log=True, prune_nans=True)

        # Fit cutoff threshold
        if eta_0 == "auto":
            eta_0 = self.__fit_cutoff_threshold(T, R)
            eta_0_diag = eta_0

        # Determine optimal axes
        T2 = prune_outliers(T)
        R2 = prune_outliers(R)
        
        xmin, xmax = T2.min(), T2.max()
        ymin, ymax = R2.min(), R2.max()

        xmin = np.sign(xmin) * np.ceil(np.abs(xmin))
        xmax = np.sign(xmax) * np.ceil(np.abs(xmax))
        ymin = np.sign(ymin) * np.ceil(np.abs(ymin))
        ymax = np.sign(ymax) * np.ceil(np.abs(ymax))

        dx = xmax - xmin
        dy = ymax - ymin
        
        if dx >= dy:
            ymax = ymin + dx

        else:
            xmax = xmin + dy

        if eta_0_diag is not None:
            dx = 0.5 * (eta_0_diag - xmax - ymin)
            xmin += dx
            xmax += dx
            ymin += dx
            ymax += dx

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

    def seismicity_rate(self, tbins, return_cumulative=False):
        """
        Get seismicity rate.

        Parameters
        ----------
        tbins : timedelta_like or sequence of datetime_like
            If `tbins` is a timedelta_like, it defines the width of each bin.
            If `tbins` is a sequence of datetime_like, it defines a monotonically increasing list of bin edges.

        Returns
        -------
        array_like
            Seismicity rate (in events/year) or cumulative seismicity (number of events).
        sequence of datetime_like
            Dates.

        """
        if isinstance(tbins, (timedelta, np.timedelta64)):
            tmin = min(self.origin_times)
            tmax = max(self.origin_times)
            tbins = np.arange(tmin, tmax, tbins, dtype="M8[ms]").tolist()

        elif isinstance(tbins, (list, tuple, np.ndarray)):
            if any(not isinstance(t, (datetime, np.datetime64)) for t in tbins):
                raise TypeError()
            tbins = tbins

        else:
            raise TypeError()

        ty = to_decimal_year(self.origin_times)
        tybins = to_decimal_year(tbins)
        hist, _ = np.histogram(ty, bins=tybins)

        out = hist.cumsum() if return_cumulative else hist / np.diff(tybins)
        t = np.array(tbins[:-1]) + 0.5 * np.diff(tbins)

        return out, t.tolist()

    @property
    def origin_times(self):
        """Return origin times."""
        return self._origin_times

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

    @property
    def years(self):
        """Return origin times in decimal years."""
        return to_decimal_year(self._origin_times)
