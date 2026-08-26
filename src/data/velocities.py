"""Velocities: differentiating trajectories, rejecting impossible readings, and
averaging the result over a spatial grid.

All velocities are in px/frame. Converting to mm/s needs `k = fps / px_per_mm`, which
the callers that work in physical units pass in.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from src.data.histograms import density_valley
from src.model.velocity_field import VelocityField

SG_WINDOW = 7               # frames per Savitzky-Golay window
SG_POLY = 2                 # polynomial degree of the fit
GRID_SIZE = 25              # px, side of a velocity-field cell
PRELIM_BINS = 120           # bins for the preliminary static/mobile split
PRELIM_SIGMA = 3            # smoothing of those bins, in bins
TUKEY_MULT = 3              # Q3 + TUKEY_MULT * IQR — the "extreme" Tukey fence


def _smooth_derivative(trajectory: pd.Series, window: int, poly: int) -> NDArray | float:
    """Differentiate one coordinate of one trajectory with a Savitzky-Golay filter.

    Instead of a two-point difference, fits p(t) = at^2 + bt + c over a sliding time
    window and takes its analytic derivative, which is far less noisy.

    Args:
        trajectory: The `x` or `y` values of a single particle, ordered by frame.
        window: Number of frames per window.
        poly: Degree of the polynomial fitted in each window.

    Returns:
        The velocity at each point, in px/frame. Returns NaN for trajectories with no
        more points than the window, which cannot be fitted; `add_velocities` drops
        those rows.
    """
    if len(trajectory) > window:
        return savgol_filter(trajectory, window, poly, deriv=1, mode="interp")
    return np.nan


def add_velocities(
    tracks: pd.DataFrame,
    sg_window: int = SG_WINDOW,
    sg_poly: int = SG_POLY,
) -> pd.DataFrame:
    """Add the velocity columns to a trajectory table.

    Track edges are NOT trimmed: the start of a fall — a pocket beginning to move —
    lives in the first frames of each track and is real signal. Edge noise is dealt
    with later, by the static/mobile threshold.

    Args:
        tracks: Filtered trajectories; needs `particle`, `frame`, `x` and `y`.
        sg_window: Number of frames per Savitzky-Golay window.
        sg_poly: Degree of the polynomial fitted in each window.

    Returns:
        The input table sorted by (particle, frame) and re-indexed from 0, with `vx`,
        `vy` and `v` (the modulus) added, all in px/frame. Trajectories too short to
        differentiate are dropped.
    """
    tracks = tracks.sort_values(["particle", "frame"]).copy()

    grouped = tracks.groupby("particle", group_keys=True)
    tracks["vx"] = grouped.x.transform(_smooth_derivative, sg_window, sg_poly)
    tracks["vy"] = grouped.y.transform(_smooth_derivative, sg_window, sg_poly)
    tracks = tracks.dropna(subset=["vx", "vy"])

    tracks["v"] = np.sqrt(tracks["vx"] ** 2 + tracks["vy"] ** 2)
    return tracks.reset_index(drop=True)

def compute_velocity_field(velocities: pd.DataFrame, grid_size: int = GRID_SIZE) -> VelocityField:
    """Average the per-particle velocities over a square grid, one field per frame.

    The grid spans the extent of the tracked beads, not the full frame, so its origin
    is `velocities.x.min()` / `velocities.y.min()` rather than the top-left pixel.
    Cells with no bead in a given frame are left at 0.

    Args:
        velocities: Per-(particle, frame) table; needs `frame`, `x`, `y`, `vx`, `vy`.
        grid_size: Side of a cell in px. Bigger cells average more beads together, so
            they are smoother but blur the boundary between the static and the moving
            regions.

    Returns:
        The raw `VelocityField`. Since the cells are already averaged, a phase filter
        cannot be applied to it afterwards — a filtered field has to be rebuilt from
        the velocities table.
    """
    x_bins = np.arange(velocities.x.min(), velocities.x.max() + grid_size, grid_size)
    y_bins = np.arange(velocities.y.min(), velocities.y.max() + grid_size, grid_size)
    nx_bins, ny_bins = len(x_bins) - 1, len(y_bins) - 1

    binned = velocities.assign(
        x_bin=pd.cut(velocities["x"], bins=x_bins, labels=False),
        y_bin=pd.cut(velocities["y"], bins=y_bins, labels=False),
    )
    cells = binned.groupby(["frame", "y_bin", "x_bin"])[["vx", "vy"]].mean()

    def dense_grid(frame_idx: int, column: str) -> NDArray[np.float64]:
        """Reshape one frame's sparse cell means into a full (ny_bins, nx_bins) grid."""
        try:
            grid = cells.loc[frame_idx][column].unstack(fill_value=0)
        except KeyError:
            return np.zeros((ny_bins, nx_bins))
        return grid.reindex(index=range(ny_bins), columns=range(nx_bins), fill_value=0).values

    frames_list = sorted(binned["frame"].unique())
    return VelocityField(
        Px=np.array([dense_grid(f, "vx") for f in frames_list]),
        Py=np.array([dense_grid(f, "vy") for f in frames_list]),
        frames_list=np.array(frames_list),
        x_bins=x_bins,
        y_bins=y_bins,
    )


def filter_velocity_outliers(
    velocities: pd.DataFrame,
    k: float,
    tukey_mult: float = TUKEY_MULT,
) -> tuple[pd.DataFrame, float]:
    """Drop the readings whose velocity is physically impossible.

    Some readings have a bead "accelerating" to thousands of mm/s in a single frame.
    The tail of the distribution has no natural break to cut at, so the cut comes from
    a standard statistical rule instead: the Tukey fence `Q3 + 3*IQR`, the same
    criterion that draws the whiskers of a boxplot.

    The fence is computed over the MOBILE readings only — including the static ones,
    which are the bulk of the data, would collapse the IQR and cut far too low. Mobile
    is isolated with a quick preliminary threshold (the valley of the raw log-velocity
    distribution); that threshold is used for nothing else.

    Args:
        velocities: Per-(particle, frame) table; needs `v` in px/frame.
        k: Conversion factor from px/frame to mm/s (`fps / px_per_mm`).
        tukey_mult: Multiplier of the IQR in the fence.

    Returns:
        A `(kept, v_cut)` pair: the surviving rows, and the cut that was applied in
        mm/s, rounded to the nearest 10.
    """
    v_pxf = velocities['v'].values
    positive = v_pxf[v_pxf > 0]
    v_static_prelim = density_valley(np.log10(positive), PRELIM_BINS, PRELIM_SIGMA) * k

    v_mms = v_pxf * k
    mobile = v_mms[v_mms > v_static_prelim]
    q1, q3 = np.percentile(mobile, [25, 75])
    v_cut = round((q3 + tukey_mult * (q3 - q1)) / 10) * 10

    return velocities[v_mms <= v_cut].copy(), float(v_cut)
