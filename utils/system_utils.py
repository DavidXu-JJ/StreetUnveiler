#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

from errno import EEXIST
from os import makedirs, path
import os
import re
import glob

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        try:
            saved_iters.append(int(fname.split("_")[-1]))
        except ValueError:
            pass
    if saved_iters:
        return max(saved_iters)
    else:
        return 0

def searchForMaxInpaintRound(dir_path):

    all_files = os.listdir(dir_path)

    pattern = re.compile(r'instance_workspace_(\d+)$')

    matching_dirs = [(f, int(pattern.match(f).group(1))) for f in all_files if pattern.match(f)]

    if matching_dirs:
        max_dir = max(matching_dirs, key=lambda x: x[1])
    else:
        return -1

    return max_dir[1]

def count_png_files(directory):
    png_files = glob.glob(os.path.join(directory, '*.png'))
    return len(png_files)