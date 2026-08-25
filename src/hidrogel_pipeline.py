from argparse import Namespace
from pathlib import Path
'''
Extracts the background from each frame in the video

It outputs the numpy.ndarray that pims gives us but with some basic preprocessing
on the array (background extraction, increase saturation, etc)

the result is save as preprocessed.tif
'''
def preprocess(args: Namespace) -> None:
    filename        = args.filepath
    starting_frame  = args.start_at_frame
    cut_top         = args.cut_top
    fps             = args.fps
    outputs_folder  = args.outputs_folder

    Path(outputs_folder, "preprocessed.tif",).touch()
    print("preprocesado terminado")

'''
    It detects the hidrogels on each frame of the video.
    It requires preprocessed.tif
    the result is save as features.csv
'''
def extract_features(args: Namespace) -> None:
    Path(args.outputs_folder, "features.csv",).touch()
    print("extract_features")

'''
    extracts velocities for each trayectory
    requires: features.csv
    saves to: velocities.csv
'''
def extract_velocities(args: Namespace) -> None:
    Path(args.outputs_folder, "velocities.csv",).touch()
    print("todo")

'''
   calculates the P velocity field it putputs a video using
   Vx, Vy and mod(V) as the RGB channels
   requires: velocities.csv
   saves to: fields.tif
'''
def calculate_velocity_field(args: Namespace) -> None:
    print("todo")


def transition_velocity_area(args: Namespace) -> None:
    print("todo")
