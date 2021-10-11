'''
Library for basic video functions
'''

# Imports
import os
import cv2
import time
import subprocess
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def SaveFrames2Video(frames, pathOut, fps=20.0, size=None):
    if os.path.splitext(pathOut)[-1] == '.gif':
        frames_images = [Image.fromarray(frame) for frame in frames]
        extraFrames = []
        if len(frames_images) > 1:
            extraFrames = frames_images[1:]
        frames_images[0].save(pathOut, save_all=True, append_images=extraFrames, format='GIF', loop=0)
    else:
        if size is None: size = (frames[0].shape[1], frames[0].shape[0])
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        for frame in frames:
            out.write(frame)
        out.release()

def FixVideoFile(pathIn, pathOut):
    COMMAND_VIDEO_CONVERT = 'ffmpeg -i \"{path_in}\" -vcodec libx264 \"{path_out}\"'
    
    if os.path.exists(pathOut):
        os.remove(pathOut)

    convert_cmd = COMMAND_VIDEO_CONVERT.format(path_in=pathIn, path_out=pathOut)
    print("Running Conversion Command:")
    print(convert_cmd + "\n")
    ConvertOutput = subprocess.getoutput(convert_cmd)

def DisplayImageSequence(imgSeq, delays, delayScale=0.1):
    delays = np.array(delays)
    delays_Norm = (delays - np.min(delays)) / (np.max(delays) - np.min(delays)) * delayScale
    frameNo = 0
    while(True):
        if delays_Norm[frameNo % len(imgSeq)] == 0.0 and ((frameNo % len(imgSeq)) < (len(imgSeq)-1)):
            frameNo += 1
            continue
        frame = imgSeq[frameNo % len(imgSeq)]
        cv2.imshow('Exec', frame)

        time.sleep(delays_Norm[frameNo % len(imgSeq)])

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frameNo += 1

def ViewImage(I):
    cv2.imshow('' ,I)
    cv2.waitKey(0)

# Driver Code