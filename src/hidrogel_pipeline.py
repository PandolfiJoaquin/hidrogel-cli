from pathlib import Path
'''
Extracts the background from each frame in the video

It outputs the numpy.ndarray that pims gives us but with some basic preprocessing
on the array (background extraction, increase saturation, etc)

the result is save as preprocessed.tif
'''
def preprocess(args):
    filename        = args.filepath
    starting_frame  = args.start_at_frame
    cut_top         = args.cut_top
    fps             = args.fps
    outputs_folder  = args.outputs_folder

    Path(outputs_folder).mkdir(parents=True, exist_ok=True)
    Path(outputs_folder, "preprocessed.tif",).touch()
    print("preprocesado terminado")

'''
    It detects the hidrogels on each frame of the video.
    It requires preprocessed.tif
    the result is save as features.pickle
'''
def extract_features(args):
    print("extract_features")

def todo(args):
    print("todo")
