"""
Generate scenarios from a folder contains multiple WAYMO data records
Author: Detian Guo
Date: 04/11/2022
"""
import argparse
import json
import re
import time
import traceback

from rich.progress import track

from logger.logger import *
from scenario_miner import ScenarioMiner
from tags_generator import TagsGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--eval_mode', action="store_true" ,help='[bool] True for evaluation mode')
parser.add_argument('--specified_file', type=str, required=False, help='specify a data to process')
eval_mode = parser.parse_args().eval_mode
specified_file = parser.parse_args().specified_file
# working directory 
# resolve() is to get the absolute path
ROOT = Path(__file__).resolve().parent.parent

# modify the following two lines to your own data and result directory
DATA_DIR = ROOT / "waymo_open_dataset" / "data" / "tf_example" / "training"

if eval_mode:
    DATA_DIR = ROOT / "waymo_open_dataset" / "data" / "eval_data" / "carla_data"

DATA_DIR_WALK = DATA_DIR.iterdir()
RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
RESULT_DIR = ROOT / "results/gp1" / RESULT_TIME
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir(exist_ok=True, parents=True)

# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, required=True, help='#file to plot.e.g.:00003')
# args = parser.parse_args()

if __name__ == '__main__':
    time_start = time.perf_counter()
    for DATA_PATH in track(DATA_DIR_WALK, description="Processing files"):
        FILE = DATA_PATH.name
        if not FILE.endswith(".pkl") or FILE.endswith(".jpg"):
            continue
        if specified_file and specified_file != FILE:
            continue
        FILENUM = re.search(r"-(\d{5})-", FILE)
        if FILENUM is not None:
            FILENUM = FILENUM.group()[1:-1]
            print(f"Processing file: {FILE}")
        else:
            print(f"File name error: {FILE}")
            continue
        result_dict = {}
        RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
        if eval_mode:
            fileprefix = FILE.split('-')[0]
        else:
            fileprefix = 'Waymo'
        RESULT_FILENAME = f'{fileprefix}_{FILENUM}_{RESULT_TIME}_tag.json'
        try:
            #   tagging
            tags_generator = TagsGenerator()
            general_info, \
            inter_actor_relation, \
            actors_activity, \
            actors_environment_element_intersection = tags_generator.tagging(DATA_DIR / FILE,FILE,eval_mode=eval_mode)
            result_dict = {
                'general_info': general_info,
                'inter_actor_relation': inter_actor_relation,
                'actors_activity': actors_activity,
                'actors_environment_element_intersection': actors_environment_element_intersection
            }
            with open(RESULT_DIR / RESULT_FILENAME, 'w') as f:
                print(f"Saving tags.")
                json.dump(result_dict, f)
            # #####  scenario categorization   #####
            # scenario_categorizer = ScenarioCategorizer(FILENUM,result_dict)
            # SC_dict = scenario_categorizer.find_SC('SC1')
            # RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_SC1.json'
            # with open(RESULT_DIR / RESULT_FILENAME,'w') as f:
            #     print(f"Saving SC1.")
            #     json.dump(SC_dict,f)
            #  solo scenario mining
            scenario_miner = ScenarioMiner()
            solo_scenarios = scenario_miner.mining(result_dict)
            RESULT_FILENAME = f'{fileprefix}_{FILENUM}_{RESULT_TIME}_solo.json'
            with open(RESULT_DIR / RESULT_FILENAME, 'w') as f:
                print(f"Saving solo scenarios.")
                json.dump(solo_scenarios, f)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"FILE:{FILENUM}.\nTag generation error:{e}")
            logger.error(f"trace:{trace}")
    ############################################################################
    # messager for finishing one data record. Comment out this if you don't use weChat
    # wechatter(f"FILE:{FILENUM} finished.")
    ############################################################################
    time_end = time.perf_counter()
    print(f"Time cost: {time_end - time_start:.2f}s.RESULT_DIR: {RESULT_DIR}")
    logger.info(f"Time cost: {time_end - time_start:.2f}s.RESULT_DIR: {RESULT_DIR}")
