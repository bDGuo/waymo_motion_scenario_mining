# imports and global setting

import os
import time
import argparse
from plotting_scenarios import plot_all_scenarios

ROOT = os.path.abspath(os.path.dirname(""))
ROOT = os.path.dirname(ROOT)
DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(""))),r"figures\scenarios\v2")


RESULTDIR = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(""))),r"results\v4")
RESULT_TIME = f'2022-11-09-09_25'


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='#file to plot.e.g.:00003')
args = parser.parse_args()

if __name__ == "__main__":
    start1 = time.perf_counter()
    FILENUM = args.file
    FILE = f"training_tfexample.tfrecord-{FILENUM}-of-01000"
    RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
    RESULT_SOLO = f'Waymo_{FILENUM}_{RESULT_TIME}_solo.json'
    # sanity check
    FILENAME = os.path.join(DATADIR,FILE)
    start2 = time.perf_counter()
    if os.path.exists(FILENAME):
        print(f"Plotting:{FILE}")
        _=plot_all_scenarios(DATADIR,FILE,FILENUM,RESULTDIR,RESULT_FILENAME,RESULT_SOLO,FIGDIR)
        
    else:
        print(f"File not found:{FILE}")
    end1 = time.perf_counter()
    print(f"Time1 elapsed:{end1-start1:2f}s")
    print(f"Time2 elapsed:{end1-start2:2f}s")
