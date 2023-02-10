"""
Called by runner.py
"""
from collections import namedtuple

# Path: utils\tags_dict.py

lo_act_dict = {
"2":'standing still',
"1":'accelerating',
"0":'cruising',
"-1":'decelerating',
"-2":'reversing',
'-5':'invalid'
}

la_act_dict = {
"1":'turning left',
"0":'going straight',
"-1":'turning right',
"-5":'invalid'
}

road_relation_dict = {
"11": 'lane changing',
"10": 'no lane change',
"3":'staying',
"2":'entering',
"1":'approaching',
"0":'leaving',
"-1":'not relative',
"-5":'invalid'
}


inter_actor_relation_dict = {
"0":"not related",
"1":"related 1",
"2":"related 2",
"3":"related 1 and 2"
}
inter_actor_position_dict = {
"0":"not related",
"1":"front",
"2":"left",
"3":"right",
"4":"back"
}

light_state_dict = {
    "-1":"Invalid",
    "0":"Unknown",
    "1":"Arrow stop",
    "2":"Arrow caution",
    "3":"Arrow go",
    "4":"Stop",
    "5":"Caution",
    "6":"Go",
    "7":"Flashing stop",
    "8":"Flashing caution"
}