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
import tensorflow as tf
from helpers.create_rect_from_file import features_description, get_parsed_carla_data
from logger.logger import *
from scenario_miner import ScenarioMiner
from tags_generator import TagsGenerator
from warnings import simplefilter
simplefilter('error')

parser = argparse.ArgumentParser()
parser.add_argument('--eval_mode', action="store_true" ,help='[bool] True for evaluation mode')
parser.add_argument('--ext_data',action="store_true" ,help='[bool] True for using external data')
parser.add_argument('--specified_file', type=str, required=False, help='specify a data to process')
parser.add_argument('--result_folder', type=str, required=False, help='result time to be categorized, e.g. 02-28-16_35')
parser.add_argument('--start_file', type=str, required=False, help='start file to process')
eval_mode = parser.parse_args().eval_mode
ext_data = parser.parse_args().ext_data
result_folder = parser.parse_args().result_folder
specified_file = parser.parse_args().specified_file
start_file = parser.parse_args().start_file
# working directory 
# resolve() is to get the absolute path
ROOT = Path(__file__).resolve().parent.parent

# modify the following two lines to your own data and result directory
DATA_DIR = ROOT / "waymo_open_dataset" / "data" / "tf_example" / "training"
if ext_data:
    DATA_DIR = Path("F:/VRU_prediction_dataset/waymo")

if eval_mode:
    DATA_DIR = ROOT / "waymo_open_dataset" / "data" / "eval_data" / "carla_data"

DATA_DIR_WALK = DATA_DIR.iterdir()


# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, required=True, help='#file to plot.e.g.:00003')
# args = parser.parse_args()

if __name__ == '__main__':
    if result_folder:
        RESULT_TIME = f"2023-{result_folder}"
    else:
        RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
    RESULT_DIR = ROOT / "results" / "gp1" / RESULT_TIME
    if not RESULT_DIR.exists():
        RESULT_DIR.mkdir(exist_ok=True, parents=True)
    time_start = time.perf_counter()
    for DATA_PATH in track(DATA_DIR_WALK, description="Processing files"):
        FILE = DATA_PATH.name
        if eval_mode and not FILE.endswith(".pkl"):
            continue
        if specified_file and specified_file != FILE:
            print(f"Skipping file: {FILE}.")
            continue
        FILENUM = re.search(r"-(\d{5})-", FILE)
        if FILENUM is not None:
            FILENUM = FILENUM.group()[1:-1]
            print(f"Processing file: {FILE}.")
        else:
            print(f"File name error: {FILE}.")
            continue
        result_dict = {}
        try:
            # parsing data
            if eval_mode:
                parsed = get_parsed_carla_data(DATA_DIR / FILE)
                fileprefix = FILE.split('-')[0]
            else:
                fileprefix = 'Waymo'
                dataset = tf.data.TFRecordDataset(DATA_DIR / FILE, compression_type='')
                for data in dataset.as_numpy_iterator():
                    parsed = tf.io.parse_single_example(data, features_description)
                    scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")
                    print(f"Processing scene: {scene_id}.")
                    result_filename = f'{fileprefix}_{FILENUM}_{scene_id}_tag.json'
                    #   tagging
                    tags_generator = TagsGenerator()
                    general_info, \
                    inter_actor_relation, \
                    actors_activity, \
                    actors_environment_element_intersection = tags_generator.tagging(parsed,FILE)
                    result_dict = {
                        'general_info': general_info,
                        'inter_actor_relation': inter_actor_relation,
                        'actors_activity': actors_activity,
                        'actors_environment_element_intersection': actors_environment_element_intersection
                    }
                    with open(RESULT_DIR / result_filename, 'w') as f:
                        print(f"Saving tags.")
                        json.dump(result_dict, f)
                    scenario_miner = ScenarioMiner()
                    solo_scenarios = scenario_miner.mining(result_dict)
                    result_filename = f'{fileprefix}_{FILENUM}_{scene_id}_solo.json'
                    with open(RESULT_DIR / result_filename, 'w') as f:
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
