from typing import Dict
import time


def exchange_key_value(old_dict: Dict) -> Dict:
    return dict([val, key] for key, val in old_dict.items())

def sleeper(sleep_time:int):
    for i in range(sleep_time):
        time.sleep(1)
        print(f"Sleeping {i+1}/{sleep_time} seconds...")