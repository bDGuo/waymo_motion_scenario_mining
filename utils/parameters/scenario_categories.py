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
class Car2CarFrontHeadon(SCBasis):
    #####   general info    #####
    SC_ID = "SC3"
    description = "Car-to-car_front_headon"
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
        "inter_actor_position": []
    }


@dataclass
class Car2PedFrontHeadon(Car2CarFrontHeadon):
    SC_ID = "SC_10"
    description = "Car-to-pedestrian_front_headon"
    guest_actor_type = ["pedestrian"]


@dataclass
class Car2BycFrontHeadon(Car2CarFrontHeadon):
    SC_ID = "SC_4"
    description = "Car-to-cyclist_front_headon"
    guest_actor_type = ["cyclist"]


@dataclass
class Car2CarCrossStraight(SCBasis):
    #####   general info    #####
    SC_ID = "SC2"
    description = "Car-to-car_crossing_straight_crossing_path"
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
class Car2PedCrossStraight(Car2CarCrossStraight):
    SC_ID = "SC_7"
    description = "Car-to-pedestrian_crossing_straight_crossing_path"
    guest_actor_type = ["pedestrian"]


@dataclass
class Car2BycCrossStraight(Car2CarCrossStraight):
    SC_ID = "SC_12"
    description = "Car-to-cyclist_crossing_straight_crossing_path"
    guest_actor_type = ["cyclist"]


@dataclass
class CarViolateTrafficLight(SCBasis):
    #####   general info    #####
    SC_ID = "SC11"
    description = "Car violates traffic light"
    source = ""
    source_file = ""
    #####   host actor  #####
    host_actor_type = ["vehicle"]
    host_actor_tag = {
        "lo_act": [],
        'la_act': [],
        "road_relation": ['staying', 'entering', 'leaving'],
        "road_type": ['controlled_lane'],
        "inter_actor_relation": [],
        "inter_actor_position": []
    }
    #####   guest actor  #####
    # skip
    #####    environment  #####
    envr_tag = {
        'light_state': ['Arrow stop', 'Stop', 'Flashing stop']
    }

@dataclass
class Car2BycPassingby(SCBasis):
    #####   general info    #####
    SC_ID = "SC13"
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
    "SC5": Car2BycFrontTurn,
    "SC8": Car2PedFrontTurn,

    "SC3": Car2CarFrontHeadon,
    "SC10": Car2PedFrontHeadon,
    "SC4": Car2BycFrontHeadon,

    "SC11": CarViolateTrafficLight,
    
    "SC2": Car2CarCrossStraight,
    "SC7": Car2PedCrossStraight,

    "SC13": Car2BycPassingby,
}
