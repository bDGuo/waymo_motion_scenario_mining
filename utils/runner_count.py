from counter import Counter
import json
import pandas as pd
import argparse
from logger.logger import *
from rich.progress import track

# working directory
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument('--result_time', type=str, required=True, help='result time to be categorized, e.g. 02-28-16_35')
args = parser.parse_args()

RESULT_TIME = args.result_time
RESULT_DIR = ROOT / "results" / "gp1" / f"2023-{RESULT_TIME}"
sc = ["SC1","SC7","SC13"]

if __name__ == '__main__':
    flag = 1
    stats = pd.Series([1, 2], index=["a", "b"])
    for f in track(RESULT_DIR.iterdir(), description="Counting tags..."):
        if not f.name.endswith("solo.json"):
            continue
        filenum = f.name.split("_")[1]
        tags = json.load(open(f, "r"))
        counter = Counter(tags)
        stats_f = pd.concat([counter.count_tag('lo_act'), counter.count_tag('la_act'),counter.count_tag('surface_street'), counter.count_tag('bike_lane'), counter.count_tag('cross_walk')])
        if flag:
            stats = stats_f.copy()
            flag = 0
        else:
            stats.loc[:,["vehicle","pedestrian","cyclist"]] += stats_f.loc[:,["vehicle","pedestrian","cyclist"]]

    flag = 1
    stats_sc = pd.Series([1, 2], index=["a", "b"])
    for sc_id in sc:
        sc_id_path = RESULT_DIR / sc_id
        for file in sc_id_path.iterdir():
            if file.suffix == '.json':
                with open(file, 'r') as f:
                    mined_sc = json.load(f)
                    counter = Counter(mined_sc)
                    if flag:
                        stats_sc = counter.count_sc()
                        flag = 0
                    else:
                        stats_sc += counter.count_sc()

    writer = pd.ExcelWriter(f'{RESULT_DIR}_count_all.xlsx')
    stats.to_excel(writer, sheet_name='tag', index=True)
    stats_sc.to_excel(writer,sheet_name='sc', index=True)
    writer.save()
    writer.close()
