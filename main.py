import argparse
import pathlib
from platform import java_ver
from src import hidrogel_pipeline
import os

AMOUNT_OF_STEPS = 5

# def validate_steps_to_run(steps_to_run)
#     if steps_to_run != sorted(steps_to_run):
#         raise ValueError("steps must be run in order")
#     if len(steps_to_run) != len(set(steps_to_run)):
#         raise ValueError("steps can only be run once per execution")
#     if any(steps_to_run, lambda step: step > AMOUNT_OF_STEPS):
#         raise ValueError(f"There are only {AMOUNT_OF_STEPS} steps in the pipeline")



def main(args):
    steps_to_run = args.run_steps
    # validate_steps_to_run(steps_to_run)
    steps = [
        (hidrogel_pipeline.preprocess,[]),
        (hidrogel_pipeline.extract_features,["preprocessed.tif"]),
        (hidrogel_pipeline.todo,[]),
        (hidrogel_pipeline.todo,[]),
        (hidrogel_pipeline.todo,[])

    ]
    pipeline = [steps[i]
        for i in range(AMOUNT_OF_STEPS)
        if (i+1) in steps_to_run]

    for step, requirement in pipeline:
        if requirement:
            check_file_exists(requirement, args.outputs_folder)
        step(args)



def check_file_exists(filenames, folder):
    for filename in filenames:
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line  tool for hidrogel videos analysis')
    _ = parser.add_argument("-f", "--filepath", required=True, help='path to the file being analized', type=pathlib.Path)
    _ = parser.add_argument("-o", "--outputs-folder", required=False, help='folder used to save the intermediate and final results. The default is to use an outputs folder', type=pathlib.Path, default="outputs/")
    _ = parser.add_argument("-fps", "--fps", required=False, help='frames per second on the video', type=pathlib.Path, default=80)
    _ = parser.add_argument("-s", "--start-at-frame", required=False, help='skip up to this frame in the video', type=int, default=0)
    _ = parser.add_argument("-ct", "--cut-top", required=False, help='cut this amount of pixels from the top '+
        'on each frame of the video', type=int, default=0)
    _ = parser.add_argument("-cb", "--cut-bottom", required=False, help='cut this amount of pixels from the bottom '+
        'on each frame of the video', type=int, default=0)
    _ = parser.add_argument("-d", "--debug", required=False, help='enables the rendering of plots during the data processing. ' +
        'Its good to check if the different steps are working correctly or to see in which step data is getting corrupted', action='store_true')
    _ = parser.add_argument(
        "--run-steps",
        type=lambda x: [int(v) for v in x.split(",")],
        help=(
            "Select which pipeline steps to run by numeric ID. The default is to run all steps in the pipeline but to help saving time when modifying or debugging the pipeline, "
            "you can run the specifics steps you want to test. Each step saves its results on a file so that when running the next step in isolation, it has all it needs to continue working from there. "
            "Each ID maps to a specific step and may require certain preconditions. "
            "Example mapping: "
            "1=Feature recognition."
            "2=Build trayectories"
            "3=Extract velocities for each trayectory"
            "4=Extract velocity field for each frame. "
            "Example: --run-steps 1,2"),
        default=[1,2,3,4,5]
)
    args = parser.parse_args()
    main(args)
