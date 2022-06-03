import os.path
import sys
import glob
import scenedetector as sd

def process_one_file(file):
    """
    Run SceneExtractor for a single file. The extracted scenes are stored in the current directory.

    Parameters:
    file (string): filename
    """

    sd.find_scenes(file)

def process_dir(path):
    """
    Run SceneExtractor for all files in a directory. The extracted scenes are stored in the current directory.

    Parameters:
    path (string): director path
    """
    files = glob.glob(path+"/*.mp4")
    for file in files:
        process_one_file(file)

def main(source):
    if os.path.isdir(source):
        process_dir(source)
    elif os.path.isfile(source):
        process_one_file(source)
    else: print("Cannot open " + source)

if __name__ == '__main__' :
    args = sys.argv[1:]
    main(args[0])
