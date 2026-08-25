import argparse
import pathlib
import plot_utils



def main(args):
    print(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line  tool for hidrogel videos analysis')
    _ = parser.add_argument("-f", "--filepath", required=True, help='path to the file being analized', type=pathlib.Path)
    _ = parser.add_argument("-o", "--outputs", required=False, help='folder used to save the intermediate and final results. The default is to use an outputs folder', type=pathlib.Path)
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
