# dataclasses for scenario categories
# Author: Detian Guo
# Date: 2023-02-21 

from dataclasses import dataclass

class SCBasis():
    """
    This is the basis for all scenario categories.
    The dictionary of host/guest_actor_tag overwritten by the child class.
    """
    #####   general info    #####
    SC_ID: str = "SC_0"
    description = "This is the basis for all scenario categories."
    source: str = ""
    source_file: str = ""
    #####   host actor  #####
    host_actor_type = []
    host_actor_tag: dict = {
        "lo_act": [],
        "la_act": [],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": [],
        "inter_actor_position": [],
        'inter_actor_heading': [],
    }
    #####   guest actor  #####
    guest_actor_type = []
    guest_actor_tag: dict = {
        "lo_act": [],
        "la_act": [],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": [],
        "inter_actor_position": [],
        'inter_actor_heading': [],
    }
    envr_tag = {
        'light_state': []
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
        "lo_act": ['accelerating', 'cruising', 'decelerating'],  # forward
        'la_act': ['turning left'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": ['estimated collision', "close proximity", "estimated collision+close proximity"],
        "inter_actor_position": ['front'],
        'inter_actor_heading': ['opposite']
    }
    #####   guest actor  #####
    guest_actor_type = ["vehicle"]
    guest_actor_tag = {
        "lo_act": ['accelerating', 'cruising', 'decelerating'],
        "la_act": ['swerving left', 'swerving right', 'going straight'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": [],
        "inter_actor_position": [],
        'inter_actor_heading': []
    }


@dataclass
class Car2PedCrossStraight(SCBasis):
    SC_ID = "SC_7"
    description = "Car-to-pedestrian_crossing_straight_crossing_path"
    guest_actor_type = ["pedestrian"]
    source = "EURO_NCAP_2023"
    source_file = "euro-ncap-aeb-lss-vru-test-protocol-v43.pdf"
    #####   host actor  #####
    host_actor_type = ["vehicle"]
    host_actor_tag = {
        "lo_act": ['accelerating', 'cruising', 'decelerating'],  # forward
        'la_act': ['swerving left', 'swerving right', 'going straight'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": ['estimated collision', 'close proximity', "estimated collision+close proximity"],
        "inter_actor_position": ['front'],
        'inter_actor_heading': ['left','right']
    }
    #####   guest actor  #####
    guest_actor_type = ["vehicle"]
    guest_actor_tag = {
        "lo_act": ['accelerating', 'cruising', 'decelerating','standing still'],
        "la_act": ['swerving left', 'swerving right', 'going straight'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": [],
        "inter_actor_position": []
    }

@dataclass
class Car2BycPassingby(SCBasis):
    #####   general info    #####
    SC_ID = "SC_13"
    description = "The car and the bicyclist are going straight with a close proximity. The cyclist is on the left or right side to the car."
    source = ""
    source_file = ""
    #####   host actor  #####
    host_actor_type = ["vehicle"]
    host_actor_tag = {
        "lo_act": ['accelerating', 'cruising', 'decelerating','standing still'],  # forward
        'la_act': ['swerving left', 'swerving right', 'going straight'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": ['close proximity', "estimated collision+close proximity"],
        "inter_actor_position": ['left','right'],
        'inter_actor_heading': ['same']
    }
    #####   guest actor  #####
    guest_actor_type = ["cyclist"]
    guest_actor_tag = {
        "lo_act": ['accelerating', 'cruising', 'decelerating','standing still'],
        "la_act": ['swerving left', 'swerving right', 'going straight'],
        "road_relation": [],
        "road_type": [],
        "inter_actor_relation": [],
        "inter_actor_position": []
    }



scenario_catalog = {
    "SC1": Car2CarFrontTurn,
    "SC7": Car2PedCrossStraight,
    "SC13": Car2BycPassingby,
}
