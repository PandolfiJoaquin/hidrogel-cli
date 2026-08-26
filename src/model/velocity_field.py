"""Value objects produced by the pipeline."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VelocityField:
    """A per-frame 2-D grid of mean velocities.

    This is the RAW field: no phase filter has been applied. Because every cell is
    already an average, it cannot be re-thresholded afterwards — a filtered field has
    to be rebuilt from the velocities table.

    Attributes:
        Px: Mean `vx` per cell, of shape (n_frames, ny_bins, nx_bins), in px/frame.
        Py: Mean `vy` per cell, same shape and units. y grows downward, so a falling
            bead has positive `vy`.
        frames_list: Source frame of each slice; `frames_list[i]` is the frame `Px[i]`
            was built from. Only frames with at least one tracked bead appear, so it
            is not necessarily contiguous.
        x_bins: Cell edges along x, in px. Length `nx_bins + 1`.
        y_bins: Cell edges along y, in px. Length `ny_bins + 1`.
    """
    Px: NDArray[np.float64]
    Py: NDArray[np.float64]
    frames_list: NDArray[np.int64]
    x_bins: NDArray[np.float64]
    y_bins: NDArray[np.float64]

    @property
    def nx_bins(self) -> int:
        """Number of cells along x."""
        return len(self.x_bins) - 1

    @property
    def ny_bins(self) -> int:
        """Number of cells along y."""
        return len(self.y_bins) - 1
