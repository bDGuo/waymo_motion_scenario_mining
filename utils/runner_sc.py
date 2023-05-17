
import json
import traceback

import argparse
from logger.logger import *
from rich.progress import track
from scenario_categorizer import ScenarioCategorizer
from multiprocessing import Pool

# working directory
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--ext_data',action="store_true" ,help='[bool] True for using external data')
parser.add_argument('--eval_mode', action="store_true" , help='enable evaluation mode')
parser.add_argument('--result_time', type=str, required=True, help='result time to be categorized, e.g., 02-28-16_35')
parser.add_argument('--num_workers', type=int, required=False, default=1, help='number of workers for multiprocessing')
args = parser.parse_args()

RESULT_TIME = args.result_time
RESULT_DIR = ROOT / "results" / "gp1" / f"2023-{RESULT_TIME}"
if args.ext_data:
    RESULT_DIR = Path("F:/results")
    RESULT_DIR = RESULT_DIR / "gp1" / f"2023-{RESULT_TIME}"
file_prefix = "Waymo" if not args.eval_mode else "Carla"

sc = ["SC1","SC7","SC13"]

def categorize_all_sc(result_dir):
    for SC_ID in track(sc,description="Categorizing..."):
        for result_file in result_dir.iterdir():
            if not result_file.name.endswith("tag.json"):
                continue
            filenum = result_file.name.split("_")[1]
            scenenum = result_file.name.split("_")[2]
            result_dict = json.load(open(result_file, 'r'))
            file_prefix = result_file.name.split("_")[0] if args.eval_mode else "Waymo"
            scenario_categorizer = ScenarioCategorizer(result_dict)
            try:
                SC_ID_dict = scenario_categorizer.find_SC(SC_ID)
                if not len(SC_ID_dict):
                    continue
                RESULT_SC_DIR = RESULT_DIR / SC_ID
                RESULT_SC_DIR.mkdir(parents=True, exist_ok=True)
                json.dump(SC_ID_dict, open(RESULT_SC_DIR / f"{file_prefix}_{filenum}_{scenenum}_{SC_ID}.json", 'w'))
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"SC:{SC_ID}.\nTag generation:{e}")
                logger.error(f"trace:{trace}")

import time

def sleeper(sleep_time:int):
    for i in range(sleep_time):
        time.sleep(1)
        print(f"Sleeping {i+1}/{sleep_time} seconds...")

if __name__ == '__main__':
    # measure run time
    # sleeper(60*60*2)
    time_start = time.perf_counter()
    categorize_all_sc(RESULT_DIR)
    time_end = time.perf_counter()
    print(f"Entire run time :{(time_end-time_start):.2f}s.")
