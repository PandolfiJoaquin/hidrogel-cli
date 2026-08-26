"""The transition regime: deriving the two velocity edges that bracket it, tagging every
reading against them, and counting where in the silo it happens.

Separating two populations by a simple density valley is biased when the populations
have very different sizes, which is the case here — the discharge comes in bursts, so
almost everything is still almost all of the time, and the valley drifts towards the
small population. Rather than fitting the mixture, both edges are obtained by MEASURING
the thing that actually matters, directly on the trajectories:

- the lower edge is the tracker's noise floor, measured on beads confirmed to be still;
- the upper edge is where beads accelerate like they are already in free fall, measured
  on beads confirmed to have started falling.

Every knob is expressed in physical time or as a fraction of the observed range, never
as a value tuned to one particular video, so they re-derive themselves on another
dataset.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from src.data.histograms import density_valley
from src.data.tracking import SEARCH_RANGE
from src.model.transition import (
    AccelPlateau,
    NoiseCeiling,
    TransitionArea,
    TransitionThresholds,
)

W_SECONDS = 0.19            # length of the stillness window, in physical time
TARGET_FPR = 0.005          # fraction of pure-noise readings allowed above the floor
TARGET_FRAC = 0.90          # fraction of the plateau at which "falling" starts
MIN_PER_BIN = 20            # velocity bins thinner than this are dropped from the profile

# Debounce lengths in PHYSICAL TIME, not frames, so they stay comparable across videos
# shot at different rates — the same convention as the stillness window above.
# At 80 fps these come out as the 2 and 8 frames the notebook used.
CONFIRM_UP_SECONDS = 0.025
CONFIRM_DOWN_SECONDS = 0.1

REGIMES = ['static', 'trans', 'fall']


def derive_noise_ceiling(
    velocities: pd.DataFrame,
    fps: float,
    w_seconds: float = W_SECONDS,
    target_fpr: float = TARGET_FPR,
) -> NoiseCeiling:
    """Measure the tracker's noise floor by isolating beads that are genuinely still.

    A bead is taken to be still over a window if its trajectory barely drifts within
    it; the velocity those beads still report is pure localisation jitter, so its high
    percentile is the level below which velocity means nothing. This is a measurement,
    not a curve fit.

    The stillness threshold itself is the valley of the trajectory-radius distribution,
    so it is not a hand-picked number either.

    Args:
        velocities: Per-(particle, frame) table; needs `particle`, `frame`, `x`, `y`
            and `v` in px/frame.
        fps: Frames per second of the source video. The window is fixed in physical
            time so it stays comparable across videos shot at different rates.
        w_seconds: Half-length of the centred stillness window, in seconds.
        target_fpr: The only real knob: the fraction of pure-noise readings accepted to
            cross the threshold by mistake.

    Returns:
        The measured `NoiseCeiling`.
    """
    window = max(1, round(w_seconds * fps))
    ordered = velocities.sort_values(['particle', 'frame'])
    grouped = ordered.groupby('particle')

    span = lambda a: a.max() - a.min()
    rolling = {'window': 2 * window + 1, 'center': True, 'min_periods': 2 * window + 1}
    x_span = grouped['x'].rolling(**rolling).apply(span, raw=True).reset_index(level=0, drop=True)
    y_span = grouped['y'].rolling(**rolling).apply(span, raw=True).reset_index(level=0, drop=True)

    local_rad = pd.Series(0.5 * np.hypot(x_span.values, y_span.values), index=ordered.index)
    valid = local_rad.notna()

    r_valley = density_valley(np.log10(local_rad[valid & (local_rad > 0)].values))

    quiet = valid & (local_rad < r_valley)
    noise_v = ordered.loc[quiet, 'v'].values
    noise_v = noise_v[noise_v > 0]

    return NoiseCeiling(
        v_static=float(np.percentile(noise_v, 100 * (1 - target_fpr))),
        r_valley=r_valley,
        local_rad=local_rad,
        window=window,
        n_quiet=int(quiet.sum()),
        n_valid=int(valid.sum()),
    )


def derive_accel_plateau(
    velocities: pd.DataFrame,
    v_static: float,
    fps: float,
    k: float,
    target_frac: float = TARGET_FRAC,
    min_per_bin: int = MIN_PER_BIN,
) -> AccelPlateau:
    """Measure where beads start accelerating like they are already falling.

    Free fall means CONSTANT acceleration, not zero acceleration — waiting for the
    latter takes far too long to be visible in a silo. So the acceleration is profiled
    against velocity over beads with a clean onset, the plateau is the median
    acceleration over the top quarter of the observed velocity range, and the upper
    edge is the first velocity at which the measured acceleration reaches
    `target_frac` of that plateau.

    Both the plateau region and the bin width are relative to the observed range rather
    than fixed in mm/s, so they re-scale by themselves on another dataset.

    Args:
        velocities: Per-(particle, frame) table; needs `particle`, `frame` and `v`.
        v_static: The lower edge in px/frame, used to decide when a bead starts moving.
        fps: Frames per second, to turn a per-frame velocity change into mm/s^2.
        k: Conversion factor from px/frame to mm/s (`fps / px_per_mm`).
        target_frac: Fraction of the plateau at which a bead counts as falling.
        min_per_bin: Velocity bins with fewer samples than this are dropped, so a
            thinly populated bin cannot set the edge.

    Warns:
        If the resulting edge is above the tracking ceiling, `search_range * k`. A bead
        that fast moves further between frames than the linker searches for it, so it
        cannot be followed as one trajectory — velocities up there come from mis-linked
        fragments rather than real motion, and an edge derived from them is an artefact.

    Returns:
        The measured `AccelPlateau`.
    """
    ordered = velocities.sort_values(['particle', 'frame']).copy()
    by_particle = ordered.groupby('particle')

    ordered['gap'] = by_particle['frame'].diff()
    ordered['v_mms'] = ordered['v'] * k
    ordered['dv_next'] = by_particle['v_mms'].shift(-1) - ordered['v_mms']
    ordered['gap_next'] = by_particle['frame'].shift(-1) - ordered['frame']

    # A clean onset: the bead was below the floor on the previous frame and above it now,
    # with no gap in the trajectory across that step.
    ordered['mv'] = ordered['v'] > v_static
    ordered['prev_mv'] = by_particle['mv'].shift(1)
    onset = ordered.loc[
        ordered['mv'] & (ordered['prev_mv'] == False) & (ordered['gap'] == 1), 'particle'
    ].unique()

    valid = ordered[
        (ordered['gap_next'] == 1) & ordered.particle.isin(onset) & (ordered['v_mms'] > 0)
    ].copy()
    valid['accel'] = valid['dv_next'] * fps

    n_bins = max(3, int(np.sqrt(len(valid))))
    valid['vbin'] = pd.cut(valid['v_mms'], np.linspace(0, valid['v_mms'].max(), n_bins))
    by_bin = valid.groupby('vbin', observed=True)['accel']
    profile, counts = by_bin.median(), by_bin.count()
    profile = profile[counts >= min_per_bin]

    centers_v = np.array([interval.mid for interval in profile.index])
    accel_s = gaussian_filter1d(profile.values, 2)

    plateau = np.median(accel_s[centers_v > np.percentile(centers_v, 75)])
    hi = centers_v[accel_s >= target_frac * plateau][0]

    trackable_max = SEARCH_RANGE * k
    if hi > trackable_max:
        warnings.warn(
            f'upper transition edge hi={hi:.0f} mm/s is above the tracking ceiling of '
            f'{trackable_max:.0f} mm/s (search_range={SEARCH_RANGE} px/frame at this fps '
            f'and calibration). Beads that fast cannot be linked frame to frame, so this '
            f'edge is built on mis-linked trajectories, not real motion.',
            stacklevel=2,
        )

    return AccelPlateau(
        hi=float(hi),
        plateau=float(plateau),
        centers_v=centers_v,
        accel_s=accel_s,
        n_particles=len(onset),
        n_samples=len(valid),
    )


def derive_thresholds(
    velocities: pd.DataFrame,
    fps: float,
    k: float,
    target_fpr: float = TARGET_FPR,
    target_frac: float = TARGET_FRAC,
) -> TransitionThresholds:
    """Derive both edges of the transition regime in one go.

    Args:
        velocities: Per-(particle, frame) table, already cleaned of outliers.
        fps: Frames per second of the source video.
        k: Conversion factor from px/frame to mm/s (`fps / px_per_mm`).
        target_fpr: Noise fraction accepted above the lower edge.
        target_frac: Fraction of the acceleration plateau defining the upper edge.

    Returns:
        The `TransitionThresholds`, carrying both edges and the position gate that the
        tagging step needs.
    """
    noise = derive_noise_ceiling(velocities, fps, target_fpr=target_fpr)
    accel = derive_accel_plateau(velocities, noise.v_static, fps, k, target_frac=target_frac)

    return TransitionThresholds(
        v_static=noise.v_static,
        hi=accel.hi,
        r_valley=noise.r_valley,
        local_rad=noise.local_rad,
        plateau=accel.plateau,
        n_quiet=noise.n_quiet,
        n_onset_particles=accel.n_particles,
    )

def debounce_states(
    raw_code: pd.Series,
    particle: pd.Series,
    order_idx: pd.Index,
    confirm_up: int,
    confirm_down: int,
) -> pd.Series:
    """Suppress state changes that are not sustained for long enough.

    A symmetric attack/release debounce, the same pattern as voice-activity detection:
    moving up a level needs `confirm_up` consecutive frames, moving down needs
    `confirm_down`. No transition is ever accepted on a single reading, in either
    direction. `confirm_up` is the shorter of the two because a real onset accelerates
    fast and is still above the threshold on the very next frame, while settling back
    down is more ambiguous.

    Each particle is debounced independently, so one bead's state never leaks into the
    next.

    Args:
        raw_code: Per-sample state, 0 / 1 / 2, before debouncing.
        particle: Trajectory id of each sample, aligned to `raw_code`.
        order_idx: Index of `raw_code` sorted by (particle, frame). The debounce is
            sequential, so the caller has to say what "consecutive" means.
        confirm_up: Consecutive frames needed to accept a higher state.
        confirm_down: Consecutive frames needed to accept a lower state.

    Returns:
        The debounced state, realigned to the original order of `raw_code`.
    """
    raw_sorted = raw_code.loc[order_idx].values
    particle_sorted = particle.loc[order_idx].values
    out_sorted = np.empty(len(raw_sorted), dtype=int)

    n = len(particle_sorted)
    start = 0
    for i in range(1, n + 1):
        if i == n or particle_sorted[i] != particle_sorted[start]:
            segment = raw_sorted[start:i]
            state = candidate = segment[0]
            run = 1
            out_segment = np.empty(len(segment), dtype=int)
            out_segment[0] = state

            for j in range(1, len(segment)):
                code = segment[j]
                if code == candidate:
                    run += 1
                else:
                    candidate, run = code, 1
                if candidate != state and run >= (confirm_up if candidate > state else confirm_down):
                    state = candidate
                out_segment[j] = state

            out_sorted[start:i] = out_segment
            start = i

    return pd.Series(out_sorted, index=order_idx).reindex(raw_code.index).astype(int)


def tag_regimes(
    velocities: pd.DataFrame,
    thresholds: TransitionThresholds,
    fps: float,
    k: float,
    confirm_up_seconds: float = CONFIRM_UP_SECONDS,
    confirm_down_seconds: float = CONFIRM_DOWN_SECONDS,
) -> pd.DataFrame:
    """Label every (particle, frame) reading as static, in transition, or falling.

    A plain velocity cut, read bead by bead, has two failure modes, and both are
    handled here:

    1. Jitter that does not displace the bead. A bead trembling in place can have a
       one-off velocity reading above the threshold without having actually gone
       anywhere. So a reading only counts as moving if the bead ALSO displaced over a
       short window — a position gate, not just a velocity gate. Trajectory edges
       without a full window are let through (fail-open), where velocity alone decides.
    2. One-frame flickers. Even past the position gate, an isolated reading is not
       enough to believe a change of state, which is what the debounce handles.

    Args:
        velocities: Per-(particle, frame) table, already cleaned of outliers. Needs
            `particle`, `frame` and `v`.
        thresholds: The two edges plus the position gate, from `derive_thresholds`.
        fps: Frames per second of the source video, to turn the debounce lengths from
            seconds into frames.
        k: Conversion factor from px/frame to mm/s (`fps / px_per_mm`).
        confirm_up_seconds: How long a rise must hold before it is accepted, in seconds.
        confirm_down_seconds: How long a fall must hold before it is accepted, in
            seconds.

    Returns:
        A copy of the input with `reg_code` (0 static, 1 transition, 2 fall) and `reg`
        (the same thing as a categorical) added.
    """
    confirm_up = max(1, round(confirm_up_seconds * fps))
    confirm_down = max(1, round(confirm_down_seconds * fps))
    tagged = velocities.copy()
    v_pxf = tagged['v'].values
    hi_pxf = thresholds.hi_pxf(k)

    local_rad = thresholds.local_rad.reindex(tagged.index)
    moved = (local_rad.fillna(np.inf) >= thresholds.r_valley).values

    raw_code = pd.Series(
        np.where(
            (v_pxf < thresholds.v_static) | ~moved, 0,
            np.where(v_pxf < hi_pxf, 1, 2),
        ),
        index=tagged.index,
    )

    order = tagged.sort_values(['particle', 'frame']).index
    tagged['reg_code'] = debounce_states(raw_code, tagged['particle'], order, confirm_up, confirm_down)
    tagged['reg'] = pd.Categorical.from_codes(tagged['reg_code'], categories=REGIMES)
    return tagged


def count_transition_per_frame(tagged: pd.DataFrame, n_frames: int) -> NDArray[np.int64]:
    """Count how many beads are in transition on each frame.

    Args:
        tagged: Table carrying the `reg` column, from `tag_regimes`.
        n_frames: Length of the series to build, so frames where nothing is in
            transition are present as zeros instead of missing.

    Returns:
        The per-frame count, of length `n_frames`.
    """
    counts = tagged[tagged['reg'] == 'trans'].groupby('frame').size()
    return counts.reindex(range(n_frames), fill_value=0).values


def compute_transition_area(
    tagged: pd.DataFrame,
    x_bins: NDArray[np.float64],
    y_bins: NDArray[np.float64],
    n_frames: int,
) -> TransitionArea:
    """Count the beads in transition per grid cell, frame by frame.

    Discretises the silo on the same grid as the velocity field, so the transition
    "blob" can be followed through the channel as the discharge advances.

    Args:
        tagged: Table carrying `frame`, `x`, `y` and `reg`, from `tag_regimes`.
        x_bins: Cell edges along x, in px, from the velocity field.
        y_bins: Cell edges along y, in px, from the velocity field.
        n_frames: Number of frames the grid should span.

    Returns:
        The `TransitionArea`, holding both the spatial grid and the per-frame total.
    """
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    cell_of = lambda values, bins: np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)

    in_transition = tagged[tagged['reg'] == 'trans']
    counts = in_transition.groupby([
        in_transition['frame'],
        cell_of(in_transition['y'], y_bins),
        cell_of(in_transition['x'], x_bins),
    ]).size()

    grid = np.zeros((n_frames, ny, nx), dtype=np.int16)
    frame_i, y_i, x_i = (counts.index.get_level_values(i) for i in range(3))
    grid[frame_i, y_i, x_i] = counts.values

    return TransitionArea(
        grid=grid,
        per_frame=count_transition_per_frame(tagged, n_frames),
        x_bins=x_bins,
        y_bins=y_bins,
    )
