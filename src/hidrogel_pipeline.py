"""The velocity pipeline, one function per step.

Each step is what main.py schedules through `--run-steps`. A step only unpacks arguments
and moves files around; the actual maths lives in `src.data`, and the value objects it
produces in `src.model`.

Every step writes its result to disk so the next one can run in isolation.
Ported from Hidrogel-Lab `notebooks/01_velocity_pipeline.ipynb`.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pims
import tifffile
from matplotlib import pyplot as plt

from .data.preprocessing import preprocess_frames
from .data.tracking import detect_features, filter_trajectories, link_trajectories
from .data.transition import compute_transition_area, derive_thresholds, tag_regimes
from .data.velocities import (
    add_velocities,
    compute_velocity_field,
    filter_velocity_outliers,
)

PREPROCESSED_FILE = "preprocessed.tif"
FEATURES_FILE = "features.csv"
VELOCITIES_FILE = "velocities.csv"
FIELD_FILE = "velocity_field.npz"
TRANSITION_FILE = "transition.csv"
TRANSITION_AREA_FILE = "transition_area.npz"

def load_frame(frames: pims.ImageSequence, i: int) -> np.ndarray:
    """Frame i como float32 con los recortes (CUT_TOP/CUT_BOTTOM) aplicados."""
    f = np.asarray(frames[i], dtype=np.float32)
    return f

def preprocess(args: Namespace) -> None:
    """Extract the background from each frame in the video.

    Reads the source video lazily with pims, drops the frames before
    `start_at_frame`, then removes the background and raises the contrast so the
    tracers are as bright as possible without saturating.

    Args:
        args: Parsed CLI arguments. Uses `filepath` (source video),
            `start_at_frame` (how many leading frames to skip), `cut_top` and
            `cut_bottom` (pixel bands to blank on every frame), and
            `outputs_folder` (where the result is written).

    Requires:
        Nothing; this is the first step of the pipeline.

    Saves:
        `preprocessed.tif` — the uint8 (n_frames, height, width) stack.
    """
    frames = pims.open(str(args.filepath))[args.start_at_frame:]

    if args.debug:

        plt.figure(figsize=(6, 8))
        plt.imshow(load_frame(frames, 0), cmap="gray")
        plt.title("Paso 1: Original Frame")
        plt.axis("off")
        plt.show()
        plt.close()
    stack = preprocess_frames(frames, args.cut_top, args.cut_bottom)

    destination = Path(args.outputs_folder, PREPROCESSED_FILE)
    if args.debug:
        plt.figure(figsize=(6, 8))
        plt.imshow(stack[args.start_at_frame], cmap="gray")
        plt.title("Frame preprocesado")
        plt.axis("off")
        plt.show()
        plt.close()
    tifffile.imwrite(destination, stack)
    print(f"preprocess: {stack.shape} -> {destination}")


def extract_features(args: Namespace) -> None:
    """Detect the hidrogels on each frame of the video.

    Args:
        args: Parsed CLI arguments. Uses `outputs_folder` to find the
            preprocessed stack and to write the result.

    Requires:
        `preprocessed.tif`.

    Saves:
        `features.csv` — one row per detection, with the sub-pixel `x`, `y`, the
        `frame` it belongs to and Trackpy's shape/brightness columns (`mass`,
        `size`, `ecc`, `signal`, `raw_mass`, `ep`).
    """

    stack = tifffile.imread(Path(args.outputs_folder, PREPROCESSED_FILE))

    features = detect_features(stack, plot_test=args.debug)

    destination = Path(args.outputs_folder, FEATURES_FILE)
    features.to_csv(destination, index=False)
    print(f"extract_features: {len(features)} features -> {destination}")


def extract_velocities(args: Namespace) -> None:
    """Extract velocities for each trayectory.

    Links the per-frame detections into trajectories, discards the ones too short
    or too static to be a falling bead, and differentiates what is left.

    The result is the FULL per-(particle, frame) distribution: no phase filter is
    applied here, so both the static and the mobile phase survive and the
    static/mobile threshold can still be chosen downstream.

    Args:
        args: Parsed CLI arguments. Uses `outputs_folder` to find the features and
            to write the result.

    Requires:
        `features.csv`.

    Saves:
        `velocities.csv` — the features table plus `particle` (trajectory id) and
        the velocities `vx`, `vy` and `v` (the modulus), all in px/frame.
    """
    features = pd.read_csv(Path(args.outputs_folder, FEATURES_FILE))
    tracks = filter_trajectories(link_trajectories(features))
    velocities = add_velocities(tracks)

    destination = Path(args.outputs_folder, VELOCITIES_FILE)
    velocities.to_csv(destination, index=False)
    print(f"extract_velocities: {velocities.particle.nunique()} particles -> {destination}")


def calculate_velocity_field(args: Namespace) -> None:
    """Calculate the P velocity field.

    Averages the per-particle velocities over a square grid, producing one 2-D
    field per frame. The rendering of that field (a video using Vx, Vy and mod(V)
    as the RGB channels) is a separate concern and lives in `plot_utils`.

    Args:
        args: Parsed CLI arguments. Uses `outputs_folder` to find the velocities
            and to write the result.

    Requires:
        `velocities.csv`.

    Saves:
        `velocity_field.npz` — `Px` and `Py`, the (n_frames, ny_bins, nx_bins)
        mean velocity components per cell, plus the `frames_list` each slice comes
        from and the `x_bins` / `y_bins` / `nx_bins` / `ny_bins` grid definition.
    """
    velocities = pd.read_csv(Path(args.outputs_folder, VELOCITIES_FILE))
    field = compute_velocity_field(velocities)

    destination = Path(args.outputs_folder, FIELD_FILE)
    np.savez_compressed(
        destination,
        Px=field.Px,
        Py=field.Py,
        frames_list=field.frames_list,
        x_bins=field.x_bins,
        y_bins=field.y_bins,
        nx_bins=field.nx_bins,
        ny_bins=field.ny_bins,
    )
    print(f"calculate_velocity_field: {field.Px.shape} -> {destination}")


def transition_velocity_area(args: Namespace) -> None:
    """Locate the beads that are in transition — starting to move, not yet falling.

    Rejects the physically impossible velocity readings, derives the two edges of the
    transition regime by measuring the tracking noise floor and the acceleration of
    beads confirmed to be falling, labels every reading against those edges, and counts
    the beads in transition per grid cell and per frame.

    Args:
        args: Parsed CLI arguments. Uses `fps` and `px_per_mm` (to express the
            thresholds in physical units) and `outputs_folder` to find the inputs and
            write the results.

    Requires:
        `velocities.csv`, and `velocity_field.npz` for the grid the counts land on.

    Saves:
        `transition.csv` — the velocities table, cleaned of outliers, plus `reg_code`
        (0 static, 1 transition, 2 fall) and `reg` per (particle, frame).
        `transition_area.npz` — `grid`, the per-(frame, cell) count of beads in
        transition, `per_frame`, the same count totalled per frame, the `x_bins` /
        `y_bins` of the grid, and the derived `v_static` / `hi` / `v_cut` thresholds.
    """
    velocities = pd.read_csv(Path(args.outputs_folder, VELOCITIES_FILE))
    field = np.load(Path(args.outputs_folder, FIELD_FILE))

    fps = float(args.fps)
    k = fps / float(args.px_per_mm)     # px/frame -> mm/s
    n_frames = int(velocities["frame"].max()) + 1

    # maybe take out to another step
    clean, v_cut = filter_velocity_outliers(velocities, k)
    print(f"(outliers cut at {v_cut:.0f} mm/s) -> {Path(args.outputs_folder, TRANSITION_AREA_FILE)}")

    thresholds = derive_thresholds(clean, fps, k)
    tagged = tag_regimes(clean, thresholds, fps, k)
    area = compute_transition_area(tagged, field["x_bins"], field["y_bins"], n_frames)

    tagged.to_csv(Path(args.outputs_folder, TRANSITION_FILE), index=False)
    np.savez_compressed(
        Path(args.outputs_folder, TRANSITION_AREA_FILE),
        grid=area.grid,
        per_frame=area.per_frame,
        x_bins=area.x_bins,
        y_bins=area.y_bins,
        v_static=thresholds.v_static * k,
        hi=thresholds.hi,
        v_cut=v_cut,
    )
    print(f"transition_velocity_area: {thresholds.v_static * k:.0f}-{thresholds.hi:.0f} mm/s ")
