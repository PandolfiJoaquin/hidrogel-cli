"""Small signal helpers shared by the threshold derivations.

The convention across the transition analysis is that a population boundary is read off
a *smoothed histogram of log10 values*: the log scale keeps the long fall-velocity tail
from swamping the low end, and the smoothing keeps binning noise from inventing extrema.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def _local_extrema(y: NDArray[np.float64]) -> tuple[NDArray[np.int64], list[str]]:
    """Find the local maxima and minima of a curve by sign changes of its derivative.

    No windows and no imposed ranges — it reports whatever is there, wherever it is.

    Args:
        y: The curve to inspect, typically an already smoothed histogram.

    Returns:
        A `(indices, kinds)` pair, where `indices` are the positions of the extrema in
        `y` and `kinds[i]` is `'max'` or `'min'` for each of them, in the same order.
    """
    d = np.sign(np.diff(y))
    idx = np.where(np.diff(d) != 0)[0] + 1
    kinds = ['max' if d[i - 1] > 0 else 'min' for i in idx]
    return idx, kinds


SIGMA_PER_BIN = 3 / 120     # smoothing of 3 bins at 120 bins, kept relative to the count


def density_valley(
    log_values: NDArray[np.float64],
    n_bins: int | None = None,
    sigma: float | None = None,
) -> float:
    """Locate the valley between two populations in a log-scaled distribution.

    Histograms `log_values`, smooths the counts, and returns the first local minimum —
    the dip that separates the two humps.

    Args:
        log_values: The log10 of the quantity being split. Must be finite; callers are
            expected to have dropped the non-positive samples before taking the log.
        n_bins: Number of histogram bins. Defaults to `sqrt(len(log_values))`, the
            convention used throughout the pipeline, so the resolution follows the
            sample size instead of being fixed for one dataset.
        sigma: Standard deviation, in bins, of the gaussian smoothing applied to the
            counts before looking for extrema. Defaults to `SIGMA_PER_BIN * n_bins`,
            which keeps the smoothing proportional to the binning.

    Returns:
        The valley position back in LINEAR space (i.e. `10 ** center`), in whatever
        units `log_values` was the log of.

    Raises:
        ValueError: If the smoothed histogram has no local minimum at all, which means
            the distribution is not bimodal and the caller's assumption does not hold.
    """
    if n_bins is None:
        n_bins = int(np.sqrt(len(log_values)))
    if sigma is None:
        sigma = SIGMA_PER_BIN * n_bins

    hist, edges = np.histogram(log_values, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(hist.astype(float), sigma)

    idx, kinds = _local_extrema(smoothed)
    if 'min' not in kinds:
        raise ValueError('no valley found: the distribution does not look bimodal')
    return float(10 ** centers[idx[kinds.index('min')]])
