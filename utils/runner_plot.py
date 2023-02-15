# imports and global setting

import os
import time
import argparse
from plotting_scenarios import plot_all_scenarios
from logger.logger import *
from pathlib import Path
from helpers.wechatter import wechatter

# working directory
ROOT = Path(__file__).parent.parent

# modify the following two lines to your own data,figures, and result directory
DATADIR = ROOT / "waymo_open_dataset/data/tf_example/training"
FIGDIR = ROOT / "figures/scenarios/v3"
RESULTDIR = ROOT / "results/v6"

parser = argparse.ArgumentParser()
parser.add_argument('--filenum', type=str, required=True, help='#file to plot.e.g.:00003')
parser.add_argument('--result_time', type=str, required=True, help='#result time to plot.e.g.:11-09-09_25')
args = parser.parse_args()

if __name__ == "__main__":
    start = time.perf_counter()
    FILENUM = args.filenum
    RESULT_TIME = f"2022-{args.result_time}"
    FILE = f"training_tfexample.tfrecord-{FILENUM}-of-01000"
    RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
    RESULT_SOLO = f'Waymo_{FILENUM}_{RESULT_TIME}_solo.json'
    # sanity check
    FILE_PATH = DATADIR / FILE
    if FILE_PATH.exists():
        print(f"Plotting:{FILE}")
        try:
            _=plot_all_scenarios(DATADIR,FILE,FILENUM,RESULTDIR,RESULT_FILENAME,RESULT_SOLO,FIGDIR)
        except Exception as e:
            ##########################################
            # messager for finishing one data record. Comment out this if you don't use wechat
            wechatter(f"Error in plotting {FILENUM}")
            ##########################################
            logger.error(f"FILE: {FILENUM}.{e}")
    else:
        print(f"File not found:{FILE}")
        logger.info(f"File not found:{FILE_PATH}")
    end = time.perf_counter()
    logger.info(f"DATA:{FILENUM}.JSON:{RESULT_SOLO}.Run time: {end-start}")

    # messager for finishing using wechat. Comment out this if you don't want to use wechat
    wechatter(f"{FILENUM} Plot finished. Run time: {end-start}")

