"""
Called by runner.py
Contains the meaning of the tags
"""

lo_act_dict = {
    "2": 'standing still',
    "1": 'accelerating',
    "0": 'cruising',
    "-1": 'decelerating',
    "-2": 'reversing',
    '-5': 'invalid'
}

la_act_dict = {
    "2": 'swerving left',
    "1": 'turning left',
    "0": 'going straight',
    "-1": 'turning right',
    '-2': 'swerving right',
    "-5": 'invalid'
}

road_relation_dict = {
    "11": 'lane changing',
    "10": 'no lane change',
    "3": 'staying',
    "2": 'entering',
    "1": 'approaching',
    "0": 'leaving',
    "-1": 'not relative',
    "-5": 'invalid'
}

inter_actor_relation_dict = {
    "0": "not related",
    "1": "estimated collision",
    "2": "close proximity",
    "3": "estimated collision+close proximity"
}
inter_actor_position_dict = {
    "-1": "unknown",
    "0": "not related",
    "1": "front",
    "2": "left",
    "3": "right",
    "4": "back"
}

inter_actor_heading_dict = {
    "-1": "unknown",
    "0": "not related",
    "1": "same",
    "2": "left",
    "3": "right",
    "4": "opposite"
}

inter_actor_lane_id_dict = {
    "0": "not related",
    "1": "same",
    "2": "different"
}

light_state_dict = {
    "-1": "Invalid",
    "0": "Unknown",
    "1": "Arrow stop",
    "2": "Arrow caution",
    "3": "Arrow go",
    "4": "Stop",
    "5": "Caution",
    "6": "Go",
    "7": "Flashing stop",
    "8": "Flashing caution"
}

TagDict = {
    'lo_act': lo_act_dict,
    'la_act': la_act_dict,
    'road_relation': road_relation_dict,
    'inter_actor_relation': inter_actor_relation_dict,
    'inter_actor_position': inter_actor_position_dict,
    'inter_actor_lane_id': inter_actor_lane_id_dict,
    'light_state': light_state_dict,
    'inter_actor_heading': inter_actor_heading_dict,
}


def reverse_k_v(org_dict):
    return {v: k for k, v in org_dict.items()}


ReTagDict = {k: reverse_k_v(v) for k, v in TagDict.items()}
