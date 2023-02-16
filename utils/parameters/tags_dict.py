"""
Called by runner.py
Contains the meaning of the tags
"""
from collections import namedtuple
from dataclasses import dataclass

@dataclass
class Tag:
    lo_act_dict:dict
    la_act_dict:dict
    road_relation_dict:dict
    inter_actor_relation_dict:dict
    inter_actor_position_dict:dict
    light_state_dict:dict

    def set_lo_act(self,lo_act):
        self.lo_act = lo_act
    def set_la_act(self,la_act):
        self.la_act = la_act
    def set_road_relation(self,road_relation):
        self.road_relation = road_relation
    def set_inter_actor_relation(self,inter_actor_relation):
        self.inter_actor_relation = inter_actor_relation
    def set_inter_actor_position(self,inter_actor_position):
        self.inter_actor_position = inter_actor_position
    def set_light_state(self,light_state):
        self.light_state = light_state

lo_act_dict = {
"2":'standing still',
"1":'accelerating',
"0":'cruising',
"-1":'decelerating',
"-2":'reversing',
'-5':'invalid'
}

la_act_dict = {
"2":'swerving left',
"1":'turning left',
"0":'going straight',
"-1":'turning right',
'-2':'swerving right',
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