from typing import Dict


def exchange_key_value(old_dict: Dict) -> Dict:
    return dict([val, key] for key, val in old_dict.items())
