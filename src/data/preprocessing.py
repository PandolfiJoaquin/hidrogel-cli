"""Background removal and contrast stretching.

Everything here streams the video frame by frame: only one frame lives in RAM at a time
until the final uint8 stack is built.

`frames` arguments are any lazy, indexable sequence of 2-D frames — in practice the
`pims` reader opened by the pipeline.
"""

import numpy as np
from numpy.typing import NDArray

CONTRAST_SAMPLES = 1000     # frames sampled when estimating the contrast limits
CONTRAST_TARGET = 240.0     # value v_max is stretched to (leaves headroom below 255)


def load_frame(frames, i: int, cut_top: int = 0, cut_bottom: int = 0) -> NDArray[np.float32]:
    """Read one frame as float32 with the top/bottom bands blanked out.

    The bands are zeroed rather than cropped so every frame keeps the original
    geometry and detected coordinates stay comparable to the raw video.

    Args:
        frames: Indexable sequence of 2-D frames.
        i: Index of the frame to read.
        cut_top: Number of pixel rows to blank at the top of the frame.
        cut_bottom: Number of pixel rows to blank at the bottom of the frame.

    Returns:
        The frame as a fresh float32 array of shape (height, width). Always a copy,
        so callers may write into it.
    """
    frame = np.asarray(frames[i], dtype=np.float32)
    if cut_top:
        frame[:cut_top, :] = 0
    if cut_bottom:
        frame[-cut_bottom:, :] = 0
    return frame


def compute_background(frames, cut_top: int = 0, cut_bottom: int = 0) -> NDArray[np.float32]:
    """Compute the static background as the per-pixel temporal minimum.

    The empty silo is dark, so the reference state of a pixel is its darkest moment
    over the whole video, not its brightest.

    Args:
        frames: Indexable sequence of 2-D frames.
        cut_top: Number of pixel rows to blank at the top of every frame.
        cut_bottom: Number of pixel rows to blank at the bottom of every frame.

    Returns:
        The background as a float32 array of shape (height, width).
    """
    background = load_frame(frames, 0, cut_top, cut_bottom)
    for i in range(1, len(frames)):
        np.minimum(background, load_frame(frames, i, cut_top, cut_bottom), out=background)
    return background


def compute_contrast_limits(
    frames,
    background: NDArray[np.float32],
    cut_top: int = 0,
    cut_bottom: int = 0,
    samples: int = CONTRAST_SAMPLES,
) -> tuple[float, float]:
    """Estimate the contrast range to stretch `frame - background` over.

    `v_min` is the background level (p50, so everything below it goes black, which is
    what actually raises the contrast) and `v_max` the typical tracer brightness
    (p99.9). Both come from a per-frame percentile plus a median across the sampled
    frames, which uses a lot of data without ever building a huge array.

    Args:
        frames: Indexable sequence of 2-D frames.
        background: Background returned by `compute_background`.
        cut_top: Number of pixel rows to blank at the top of every frame.
        cut_bottom: Number of pixel rows to blank at the bottom of every frame.
        samples: Approximate number of frames to sample; the stride is derived from it.

    Returns:
        The `(v_min, v_max)` pair, in the units of the background-subtracted frame.
    """
    n_frames = len(frames)
    stride = max(1, n_frames // samples)

    lows, highs = [], []
    for i in range(0, n_frames, stride):
        signal = load_frame(frames, i, cut_top, cut_bottom) - background
        lows.append(np.percentile(signal, 50))
        highs.append(np.percentile(signal, 99.9))

    return float(np.median(lows)), float(np.median(highs))


def normalize_frames(
    frames,
    background: NDArray[np.float32],
    v_min: float,
    v_max: float,
    cut_top: int = 0,
    cut_bottom: int = 0,
) -> NDArray[np.uint8]:
    """Subtract the background and stretch the result to uint8, frame by frame.

    Stretching the dynamic range makes the tracers as bright as possible without
    saturating them and losing their gaussian shape, which is what Trackpy needs for
    sub-pixel accuracy. Subtracting a min-background already erases the persistent
    reflections, so no extra mask is needed.

    Args:
        frames: Indexable sequence of 2-D frames.
        background: Background returned by `compute_background`.
        v_min: Level mapped to 0; everything below it is clipped to black.
        v_max: Level mapped to `CONTRAST_TARGET`.
        cut_top: Number of pixel rows to blank at the top of every frame.
        cut_bottom: Number of pixel rows to blank at the bottom of every frame.

    Returns:
        The whole stack as uint8, of shape (n_frames, height, width). This is the one
        array in the module that is fully materialised in RAM.
    """
    n_frames = len(frames)
    height, width = np.asarray(frames[0]).shape
    scale = CONTRAST_TARGET / (v_max - v_min)

    normalized = np.empty((n_frames, height, width), dtype=np.uint8)
    buf = np.empty((height, width), dtype=np.float32)

    for i in range(n_frames):
        np.subtract(load_frame(frames, i, cut_top, cut_bottom), background, out=buf)
        buf -= v_min
        buf *= scale
        np.clip(buf, 0, 255, out=buf)
        normalized[i] = buf

    return normalized


def preprocess_frames(frames, cut_top: int = 0, cut_bottom: int = 0) -> NDArray[np.uint8]:
    """Run the whole preprocessing chain: background removal + contrast stretch.

    Convenience wrapper over `compute_background`, `compute_contrast_limits` and
    `normalize_frames`, which is what the pipeline step calls.

    Args:
        frames: Indexable sequence of 2-D frames.
        cut_top: Number of pixel rows to blank at the top of every frame.
        cut_bottom: Number of pixel rows to blank at the bottom of every frame.

    Returns:
        The cleaned, contrasted stack as uint8, of shape (n_frames, height, width).
    """
    background = compute_background(frames, cut_top, cut_bottom)
    v_min, v_max = compute_contrast_limits(frames, background, cut_top, cut_bottom)
    return normalize_frames(frames, background, v_min, v_max, cut_top, cut_bottom)
