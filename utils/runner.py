"""
Generate scenarios from a folder contains multiple WAYMO data records
Author: Detian Guo
Date: 04/11/2022
"""
import os
import time
import re
import json
import argparse
from tags_generation import generate_tags
from mining_scenarios import mine_solo_scenarios
from rich.progress import track
from logger.logger import *

ROOT = os.path.abspath(os.path.dirname("__file__"))
ROOT = os.path.dirname(ROOT)
DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
DATADIR_WALK = os.walk(DATADIR)

RESULTDIR = os.path.join(ROOT,r"results\v4")
RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M",time.localtime())
FIGUREDIR = os.path.join(ROOT,r"figures\scenarios")

# parameters default setting
# parameter for estimation of the actor approaching a static element
TTC_1 = 5
# parameter for estimation of two actors' interaction
TTC_2 = 9
# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, required=True, help='#file to plot.e.g.:00003')
# args = parser.parse_args()

if __name__ == '__main__':
    time_start = time.perf_counter()
    for root, dirs, file_list in DATADIR_WALK:
        for FILE in track(file_list,description="Processing files"):
            FILENUM = re.search(r"-(\d{5})-",FILE)
            if FILENUM is not None:
                FILENUM = FILENUM.group()[1:-1]
                print(f"Processing file: {FILE}")
            else:
                print(f"File name error: {FILE}")
                continue    
            result_dict = {}
            RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
            try:
                actors_list,\
                inter_actor_relation,\
                actors_activity,\
                actors_static_element_intersection = generate_tags(DATADIR,FILE)
                result_dict = {
                'actors_list':actors_list,
                'inter_actor_relation':inter_actor_relation,
                'actors_activity':actors_activity,
                'actors_static_element_intersection':actors_static_element_intersection
                }
            except Exception as e:
                logger.error(f"Tag generagtion: {e}")
            with open(os.path.join(RESULTDIR,RESULT_FILENAME),'w') as f:
                json.dump(result_dict,f)
            # with open(os.path.join(RESULTDIR,RESULT_FILENAME.replace('.json','.pkl')),'wb') as f:
            #     pickle.dump(result_dict,f)
            solo_scenarios = mine_solo_scenarios(result_dict)
            RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_solo.json'
            with open(os.path.join(RESULTDIR,RESULT_FILENAME),'w') as f:
                json.dump(solo_scenarios,f)
    time_end = time.perf_counter()
    print(f"Time cost: {time_end-time_start:.2f} s" )

