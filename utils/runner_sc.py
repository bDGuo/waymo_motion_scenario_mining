
import json
import traceback

import argparse
from logger.logger import *
from parameters.scenario_categories import scenario_catalog
from scenario_categorizer import ScenarioCategorizer

# working directory
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', help='result file name prefix, e.g. PedestrianCrossing')
parser.add_argument('--result_time', type=str, required=True, help='result time to be categorized, e.g. 2023-02-28-16_35')
args = parser.parse_args()

RESULT_TIME = args.result_time
for i in range(20):
    FILE = f"0000{i}" if i < 10 else f"000{i}"
    RESULT_DIR = ROOT / "results/gp1" / RESULT_TIME
    RESULT_FILE = RESULT_DIR / f"{args.prefix}_{FILE}_{RESULT_TIME}_tag.json"
    result_dict = json.load(open(RESULT_FILE, 'r'))
    scenario_categorizer = ScenarioCategorizer(FILE, result_dict)
    for SC_ID in ["SC1","SC7","SC13"]:
        try:
            SC_ID_dict = scenario_categorizer.find_SC(SC_ID)
            if not len(SC_ID_dict):
                continue
            RESULT_SC_DIR = RESULT_DIR / SC_ID
            RESULT_SC_DIR.mkdir(parents=True, exist_ok=True)
            json.dump(SC_ID_dict, open(RESULT_SC_DIR / f"{args.prefix}_{FILE}_{RESULT_TIME}_{SC_ID}.json", 'w'))
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"SC:{SC_ID}.\nTag generation:{e}")
            logger.error(f"trace:{trace}")

