"""Value objects of the transition-velocity analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class NoiseCeiling:
    """The tracker's noise floor, measured on beads confirmed to be still.

    Attributes:
        v_static: The floor, in px/frame. Velocity below this is indistinguishable
            from localisation jitter.
        r_valley: Trajectory radius, in px, below which a bead counts as locally
            still — the valley of the radius distribution.
        local_rad: Per-sample trajectory radius in px, aligned to the index of the
            table this was derived from. NaN at the edges of a trajectory, where the
            centred window is not full.
        window: Half-length of the centred stillness window, in frames.
        n_quiet: Number of samples that were locally still — the sample the floor was
            measured on.
        n_valid: Number of samples with a full window, i.e. those eligible at all.
    """
    v_static: float
    r_valley: float
    local_rad: pd.Series
    window: int
    n_quiet: int
    n_valid: int


@dataclass
class AccelPlateau:
    """The acceleration profile of beads confirmed to be falling.

    Attributes:
        hi: First velocity, in mm/s, whose measured acceleration reaches the target
            fraction of the plateau. The upper edge of the transition regime.
        plateau: The constant acceleration of a falling bead, in mm/s^2.
        centers_v: Velocity at the centre of each profile bin, in mm/s.
        accel_s: Smoothed median acceleration in each bin, in mm/s^2. Same length as
            `centers_v`; the two together are the profile `hi` was read off.
        n_particles: Number of beads with a clean onset.
        n_samples: Number of consecutive frame pairs the profile was built from.
    """
    hi: float
    plateau: float
    centers_v: NDArray[np.float64]
    accel_s: NDArray[np.float64]
    n_particles: int
    n_samples: int


@dataclass
class TransitionThresholds:
    """The two velocity edges that bracket the transition regime.

    Both edges are MEASURED off the trajectories rather than fitted to the velocity
    distribution: `v_static` is the tracker's own noise floor, and `hi` is the velocity
    at which beads are observed to accelerate like they are already falling. Everything
    between the two is a bead in transition — starting to move, but not yet in free
    fall.

    Attributes:
        v_static: Lower edge, in px/frame, so it compares directly against the `v`
            column. Readings below it are indistinguishable from localisation jitter.
        hi: Upper edge, in mm/s. Above it a bead counts as falling.
        r_valley: Trajectory radius, in px, below which a bead counts as locally still.
            Reused as a position gate when tagging.
        local_rad: Per-sample trajectory radius in px, aligned to the index of the table
            the thresholds were derived from. NaN at the edges of a trajectory, where
            the centred window is not full.
        plateau: The constant acceleration of a falling bead, in mm/s^2, that `hi` was
            read off.
        n_quiet: Number of samples that were locally still, i.e. the sample the noise
            floor was measured on.
        n_onset_particles: Number of beads with a clean onset, i.e. the sample the
            acceleration profile was measured on.
    """
    v_static: float
    hi: float
    r_valley: float
    local_rad: pd.Series
    plateau: float
    n_quiet: int
    n_onset_particles: int

    def hi_pxf(self, k: float) -> float:
        """The upper edge in px/frame, the same units as `v_static` and the `v` column.

        Args:
            k: Conversion factor from px/frame to mm/s (`fps / px_per_mm`).

        Returns:
            `hi` divided by `k`.
        """
        return self.hi / k


@dataclass
class TransitionArea:
    """Where and how many beads are in transition, over time.

    Attributes:
        grid: Count of transition beads per (frame, y cell, x cell), of shape
            (n_frames, ny_bins, nx_bins). Counts are small — 0 to about 3 — since a
            cell only holds a bead or two.
        per_frame: Total number of beads in transition on each frame, of length
            n_frames. Index `i` is frame `i`, including the frames where nothing is in
            transition.
        x_bins: Cell edges along x, in px. Length `nx_bins + 1`.
        y_bins: Cell edges along y, in px. Length `ny_bins + 1`.
    """
    grid: NDArray[np.int16]
    per_frame: NDArray[np.int64]
    x_bins: NDArray[np.float64]
    y_bins: NDArray[np.float64]
