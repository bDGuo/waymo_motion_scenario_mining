# imports and global setting
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import time
from rich.progress import track
import pickle
from collections import namedtuple,OrderedDict
from plotting_scenarios import plot_all_scenarios
from create_rect_from_file import rect_object_creator
from helpers.diverse_plot import create_figure_and_axes

ROOT = os.path.abspath(os.path.dirname(""))
ROOT = os.path.dirname(ROOT)
DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(""))),r"figures\scenarios\v2")
FILENUM = "00003"

FILE = f"training_tfexample.tfrecord-{FILENUM}-of-01000"

RESULTDIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(""))),r"results\v4")
RESULT_TIME = f'2022-11-09-09_25'
RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
RESULT_SOLO = f'Waymo_{FILENUM}_{RESULT_TIME}_solo.json'

if __name__ == "__main__":
    _=plot_all_scenarios(DATADIR,FILE,FILENUM,RESULTDIR,RESULT_FILENAME,RESULT_SOLO,FIGDIR)