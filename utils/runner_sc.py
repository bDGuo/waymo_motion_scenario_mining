
import json
import traceback

import argparse
from logger.logger import *
from parameters.scenario_categories import scenario_catalog
from scenario_categorizer import ScenarioCategorizer

# working directory
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--eval_mode', action="store_true" , help='enable evaluation mode')
parser.add_argument('--result_time', type=str, required=True, help='result time to be categorized, e.g. 02-28-16_35')
args = parser.parse_args()

RESULT_TIME = args.result_time
RESULT_DIR = ROOT / "results/gp1" / f"2023-{RESULT_TIME}"
file_prefix = "Waymo" if not args.eval_mode else "Carla"
for file in RESULT_DIR.iterdir():
    if not file.name.endswith("tag.json"):
        continue
    filenum = file.name.split("_")[1]
    print(file.name)
    RESULT_FILE = file
    result_dict = json.load(open(RESULT_FILE, 'r'))
    file_prefix = file.name.split("_")[0] if args.eval_mode else "Waymo"
    scenario_categorizer = ScenarioCategorizer(filenum, result_dict)
    for SC_ID in ["SC1","SC7","SC13"]:
        try:
            SC_ID_dict = scenario_categorizer.find_SC(SC_ID)
            if not len(SC_ID_dict):
                continue
            RESULT_SC_DIR = RESULT_DIR / SC_ID
            RESULT_SC_DIR.mkdir(parents=True, exist_ok=True)
            json.dump(SC_ID_dict, open(RESULT_SC_DIR / f"{file_prefix}_{filenum}_{RESULT_TIME}_{SC_ID}.json", 'w'))
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"SC:{SC_ID}.\nTag generation:{e}")
            logger.error(f"trace:{trace}")

