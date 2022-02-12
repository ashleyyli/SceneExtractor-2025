import os
import sys
import decord

def process_one_file(file):
    #vr_full = decord.VideoReader(video_path, ctx=decord.cpu(0))   
    # Call ClassTranscribe  https://github.com/classtranscribe/WebAPI/blob/staging/PythonRpcServer/scenedetector.py

def process_dir(path):
    files = glob.glob(path+"/*.mp4")
    for file in files:
        process_one_file(file)

def main():
    if os.isdir(source):
        process_dir(source)
    elif os.isfile(source):
        process_one_file(source)
    else: print("Cannot open " + source)

if __name__ == '__main__' :
    main()
