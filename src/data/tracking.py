"""Tracer detection and trajectory building.

The default parameters follow Guido's thesis (§4.3.1, TrackMate LoG).
"""

import numpy as np
import pandas as pd
import trackpy as tp
from matplotlib import pyplot as plt
from numpy.typing import NDArray

DIAMETER = 9                # px, expected tracer diameter (must be odd)
MINMASS = 450               # integrated brightness cutoff; drops low-mass noise
SEARCH_RANGE = 9            # px, ~1 diameter; larger values confuse neighbouring tracks
MEMORY = 4                  # frames a particle may vanish for and still be the same track
MIN_TRACK_LENGTH = 10       # Guido §4.3.1.3: keep trajectories of >=10 frames
MIN_FALL = 20               # px of net downward displacement required over a track


def detect_features(
    stack: NDArray[np.uint8],
    diameter: int = DIAMETER,
    minmass: float = MINMASS,
    plot_test: bool = False,
) -> pd.DataFrame:
    """Locate the tracers on every frame of a preprocessed stack.

    Args:
        stack: Preprocessed uint8 stack of shape (n_frames, height, width).
        diameter: Expected tracer diameter in px. Must be odd; Trackpy uses it as the
            size of its band-pass, so it should match the beads rather than be
            generous.
        minmass: Minimum integrated brightness for a detection to be kept. This is the
            knob that separates real beads from low-mass noise.

    Returns:
        One row per detection, with the sub-pixel `x` / `y`, the `frame` it belongs to
        and Trackpy's shape and brightness columns.
    """

    if plot_test:
        test_features = tp.locate(stack[100], diameter, minmass=minmass)
        plt.figure(figsize=(14, 14))
        tp.annotate(test_features, stack[100])
        plt.close()
    tp.quiet()
    return tp.batch(stack, diameter=diameter, minmass=minmass)


def link_trajectories(
    features: pd.DataFrame,
    search_range: int = SEARCH_RANGE,
    memory: int = MEMORY,
) -> pd.DataFrame:
    """Join per-frame detections into trajectories.

    Uses a nearest-velocity predictor, so a bead already falling is looked for where
    its current velocity says it should be, not where it last was.

    Args:
        features: Detections as returned by `detect_features`; needs `x`, `y`, `frame`.
        search_range: Maximum displacement in px between consecutive frames. Roughly
            one diameter — raising it makes neighbouring tracks swap identity.
        memory: How many frames a bead may go undetected and still be reconnected to
            the same trajectory.

    Returns:
        The features table plus a `particle` column holding the trajectory id.
    """
    predictor = tp.predict.NearestVelocityPredict()
    return predictor.link_df(features, search_range=search_range, memory=memory)


def filter_trajectories(
    linked: pd.DataFrame,
    min_track_length: int = MIN_TRACK_LENGTH,
    min_fall: float = MIN_FALL,
) -> pd.DataFrame:
    """Drop the stubs and the tracers that never actually fall.

    Args:
        linked: Trajectories as returned by `link_trajectories`.
        min_track_length: Minimum number of frames a trajectory must span.
        min_fall: Minimum net downward displacement in px, measured between the first
            and the last point of a trajectory. Since y grows downward, this discards
            beads stuck to the wall that only wander around their starting point
            without ever going anywhere.

    Returns:
        The surviving rows, re-indexed from 0.
    """
    tracks = tp.filter_stubs(linked, min_track_length)
    tracks = tracks.reset_index(drop=True)  # filter_stubs leaves 'frame' as index and column

    ends = tracks.sort_values(["particle", "frame"]).groupby("particle")["y"].agg(["first", "last"])
    falling = ends.index[(ends["last"] - ends["first"]) > min_fall]
    return tracks[tracks.particle.isin(falling)].reset_index(drop=True)
