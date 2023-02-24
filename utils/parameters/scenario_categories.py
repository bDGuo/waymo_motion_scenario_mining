# dataclasses for scenario categories
# Author: Detian Guo
# Date: 2023-02-21 

from dataclasses import dataclass

class SCBasis:
    """
    This is the basis for all scenario categories.
    The dictionary of host/guest_actor_tag overwritten by the child class.
    """
    #####   general info    #####
    SC_ID : str = "SC_0"
    description = "This is the basis for all scenario categories."
    source:str = "None"
    source_file:str = "None"
    #####   host actor  #####
    host_actor_type : list = ["None"]
    host_actor_tag : dict = {
    "lo_act" : [],
    "la_act" : [],
    "road_relation" : [],
    "road_type" : [],
    "inter_actor_relation" : [],
    "inter_actor_position" : [],
    'inter_actor_v_dir':[],
    }
    #####   guest actor  #####
    guest_actor_type : list = ["None"]
    guest_actor_tag : dict = {
    "lo_act" : [],
    "la_act" : [],
    "road_relation" : [],
    "road_type" : [],
    "inter_actor_relation" : [],
    "inter_actor_position" : []
    }

@dataclass
class Car2CarFrontTurn(SCBasis):
    """
    value dimensions:
    [..., ] -> the options of tag
    [[..., ], [..., ]] -> the consequent options of tags
    """
    #####   general info    #####
    SC_ID = "SC1"
    description = "Car-to-car_front_turn_across_path"
    source = "EURO_NCAP_2023"
    source_file = "euro-ncap-aeb-lss-vru-test-protocol-v43.pdf"
    #####   host actor  #####
    host_actor_type = ["vehicle"]
    host_actor_tag = {
    "lo_act" : ['accelearting','cruising','decelerating'], # forward
    'la_act' : ['turning left','turning right'],
    "road_relation" : [],
    "inter_actor_relation" : ['estimated_collision',"estimated collision+close proximity"],
    "inter_actor_position" : ['left','right','front']    
    }
    #####   guest actor  #####
    guest_actor_type = ["vehicle"]
    guest_actor_tag = {
    "lo_act" : ['accelearting','cruising','decelerating'],
    "la_act" : ['swerving left','swerving right','going straight']
    }

@dataclass
class Car2BycFrontTurn(Car2CarFrontTurn):
    """
    Inherit from Car2CarFrontTurn
    """
    #####   general info    #####
    SC_ID = "SC5"
    #####   guest actor  #####
    guest_actor_type = ["cyclist"]

@dataclass
class Car2PedFrontTurn(Car2CarFrontTurn):
    """
    Inherit from Car2CarFrontTurn
    """
    #####   general info    #####
    SC_ID = "SC8"
    #####   guest actor  #####
    guest_actor_type = ["pedestrian"]

@dataclass
class Car2CarFollowing(SCBasis):
    #####   general info    #####
    SC_ID = "SC4"
    description = "Car-to-car_following"
    source = "EURO_NCAP_2023"
    source_file = "euro-ncap-aeb-lss-vru-test-protocol-v43.pdf"
    #####   host actor  #####
    host_actor_type = ["vehicle"]
    host_actor_tag = {
    "lo_act" : ['accelearting','cruising','decelerating'], # forward
    'la_act' : ['going straight'],
    "road_relation" : [],
    "inter_actor_relation" : ['estimated_collision',"estimated collision+close proximity"],
    "inter_actor_position" : ['front']    
    }
    #####   guest actor  #####
    guest_actor_type = ["vehicle"]
    guest_actor_tag = {
    "lo_act" : ['accelearting','cruising','decelerating'],
    "la_act" : ['going straight']
    }
    ...

ScenarioCatelog={
    "SC1":Car2CarFrontTurn,
    "SC5":Car2BycFrontTurn,
    "SC8":Car2PedFrontTurn
}
