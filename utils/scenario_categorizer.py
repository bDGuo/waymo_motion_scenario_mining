from typing import List

import numpy as np

from parameters.scenario_categories import scenario_catalog
from parameters.tags_dict import *
from logger.logger import *

class ScenarioCategorizer:
    """
    Categorizes the scenarios into 11 categories.
    """

    def __init__(self, result_dict: dict):
        self.actors_list = result_dict['general_info']['actors_list']
        self.inter_actor_relation = result_dict['inter_actor_relation']
        self.actors_activity = result_dict['actors_activity']
        self.actors_environment_element_intersection = result_dict['actors_environment_element_intersection']
        self.scenario_catalog = scenario_catalog

    def find_SC(self, scenario_category_ID: str):
        """
        Input: Scenario category ID, e.g. SC1
        Output: actor ID, start time and end time
        """
        SC = self.scenario_catalog[scenario_category_ID]
        SC_result = {}
        SC_count = 0
        for host_actor_type in SC.host_actor_type:
            for host_actor_id in self.actors_list[host_actor_type]:
                time_stamp = self._check_actor_activity(SC, host_actor_type, host_actor_id, host=True)
                #####   encode host environment relation   #####
                time_stamp *= self._check_actor_envr_relation(SC, host_actor_type, host_actor_id, time_stamp)
                #  early jump to the next host actor
                if not np.any(np.where(time_stamp == 1)[0]):
                    continue
                if not len(SC.guest_actor_type):
                    #####   result #####
                    SC_count += 1
                    time_stamp_result = np.where(time_stamp == 1)[0]
                    SC_result[SC_count] = {
                        'SC_ID': scenario_category_ID,
                        'host_actor': host_actor_id,
                        'guest_actor': "None",
                        'envr_type': SC.host_actor_tag['road_type'] if len(SC.host_actor_tag['road_type']) else "None",
                        'time_stamp': time_stamp_result.tolist()
                    }
                    continue
                #####   check guest actor   #####
                for guest_actor in self.inter_actor_relation[f'{host_actor_type}_{host_actor_id}'].keys():
                    time_stamp_g = np.zeros_like(time_stamp)
                    #####   encode inter_actor_relation and position  #####                    
                    relation_tag_encoded_h = self.tag_encoder(SC,
                                                              self.inter_actor_relation[
                                                                   f'{host_actor_type}_{host_actor_id}'][guest_actor][
                                                                   'relation'],
                                                               'inter_actor_relation')
                    position_tag_encoded_h = self.tag_encoder(SC,
                                                              self.inter_actor_relation[
                                                                   f'{host_actor_type}_{host_actor_id}'][guest_actor][
                                                                   'position'],
                                                               'inter_actor_position')
                    heading_tag_encoded_h = self.tag_encoder(SC,
                                                           self.inter_actor_relation[
                                                                f'{host_actor_type}_{host_actor_id}'][guest_actor][
                                                                'heading'],
                                                            'inter_actor_heading')
                    time_stamp_g = heading_tag_encoded_h * relation_tag_encoded_h * position_tag_encoded_h
                   #   early jump to the next guest actor
                    if not np.any(np.where(time_stamp_g == 1)[0]):
                        continue
                    #####    guest_actor   #####
                    guest_actor_type, guest_actor_id = guest_actor.split('_')
                    #####   check guest_actor type    #####
                    if (guest_actor_type in SC.guest_actor_type) is False:
                        continue
                    #####   encode guest_actor lo_act    #####
                    lo_tag_encoded_g = self.tag_encoder(SC, self.actors_activity[guest_actor_type][
                        f'{guest_actor}_activity']['lo_act'], 'lo_act', host=False)
                    #####   encode guest_actor la_act    #####
                    la_tag_encoded_g = self.tag_encoder(SC, self.actors_activity[guest_actor_type][
                        f'{guest_actor}_activity']['la_act'], 'la_act', host=False)
                    time_stamp_g *= lo_tag_encoded_g * la_tag_encoded_g
                    #  early jump to the next guest actor
                    if not np.any(np.where(time_stamp * time_stamp_g == 1)[0]):
                        continue
                    else:
                    #####   result #####
                        SC_count += 1
                        time_stamp_result = np.where(time_stamp * time_stamp_g == 1)[0]
                        SC_result[SC_count] = {
                            'SC_ID': scenario_category_ID,
                            'host_actor': host_actor_id,
                            'guest_actor': guest_actor_id,
                            'envr_type': SC.host_actor_tag['road_type'] if len(SC.host_actor_tag['road_type']) else "None",
                            'time_stamp': time_stamp_result.tolist()
                        }
        return SC_result

    def _check_actor_envr_relation(self, SC, actor_type: str, actor_id: str, time_stamp, host: bool = True):
        #####   encode host environment relation   #####
        actor_constraint = SC.host_actor_tag if host else SC.guest_actor_tag
        if not len(actor_constraint['road_type']):
            return time_stamp
        temp_time_stamp = np.zeros_like(time_stamp)
        for road_type in actor_constraint['road_type']:
            if self.actors_environment_element_intersection[actor_type][f'{actor_type}_{actor_id}'].get(
                    road_type) is None:
                continue
            elif road_type == 'controlled_lane':
                temp_time_stamp += self._check_actor_light_relation(SC, actor_type, actor_id, road_type, host=host)
            else:
                temp_time_stamp += self.tag_encoder(SC, self.actors_environment_element_intersection[actor_type][
                    f'{actor_type}_{actor_id}'][road_type]['relation'], 'road_relation', host=host)
        return np.where(temp_time_stamp > 0, 1, 0)

    def _check_actor_light_relation(self, SC, actor_type: str, actor_id: str, envr_type: str, host: bool = True):
        # encoding relation of actor and controlled lane
        relation = self.tag_encoder(SC, self.actors_environment_element_intersection[actor_type][
            f'{actor_type}_{actor_id}']['controlled_lane']['relation'], 'road_relation', host=host)
        # encoding light state of the controlled lane
        light_state = self.tag_encoder(SC, self.actors_environment_element_intersection[actor_type][
            f'{actor_type}_{actor_id}']['controlled_lane']['light_state'], 'light_state', host=host)
        return relation * light_state

    def _check_actor_activity(self, SC, actor_type: str, actor_id: str, host: bool = True):
        #####   encode inter_actor_relation and position  #####                    
        #####   encode host lo_act    #####
        lo_tag_encoded_h = self.tag_encoder(SC, self.actors_activity[actor_type][f'{actor_type}_{actor_id}_activity'][
            'lo_act'], 'lo_act', host=host)
        #####   encode host la_act    #####
        la_tag_encoded_h = self.tag_encoder(SC, self.actors_activity[actor_type][f'{actor_type}_{actor_id}_activity'][
            'la_act'], 'la_act', host=host)
        return lo_tag_encoded_h * la_tag_encoded_h

    def _check_actor_lane_id(self, SC, host_actor_type: str, host_actor_id: str, guest_actor_type: str,
                             guest_actor_id: str):
        # check if the guest actor is in the same lane as the host actor
        # TODO
        #####    environment  #####
        envr_tag = {
            'lane_id': ['same']
        }
        ...

    def tag_encoder(self, SC, encoding_tag: List, tag_type: str, host: bool = True):
        """
        Encoding of the tags according to the needed value(s)
        The encoded tag is a list of 0 and 1, 
        where 1 represents the needed value.
        Different types of encoded tag, e.g. lo_act, la_act, can be element-wise
        multiplied to get the final encoded tag which results in the scenario category. 
        ----------------
        Input:
        SC : scenario category(type: class) for the SC_ID
        encoding_tag : the tag to be encoded (type: list) from the result_dict
        tag_type : the tag type required in the scenario category,
        described in the keys of the attribute dict in scenario_categories.py
        host: whether the actor is host or guest
        ----------------
        Output: (flag, encoded tag)
        """
        actor_constraint = SC.host_actor_tag if host else SC.guest_actor_tag
        if tag_type in actor_constraint.keys():
            tag = actor_constraint[tag_type]
        else:
            tag = SC.envr_tag[tag_type]
        if not len(tag):
            return np.ones_like(encoding_tag)

        # SC tag constraints sanity check
        assert tag_type in TagDict.keys(), f"Tag type {tag_type} not found in TagDict"
        for t in tag:
            assert t in TagDict[tag_type].values(), f"Tag value {t} not found in TagDict"

        value = [float(k) for (k, v) in TagDict[tag_type].items() if v in tag]

        tag_array = np.array(encoding_tag)
        value_array = np.array(value)
        encoded_tag = np.zeros_like(tag_array)
        encoding_index = np.array([])
        for v in value_array:
            encoding_index = np.append(encoding_index, np.where(tag_array == v)[0]).astype(int)
        encoded_tag[encoding_index] = 1
        return encoded_tag

    def _tag_segmentation(self, tag: list, value: list):
        """
        Segmenting of the tags according to the needed value(s)
        ----------------
        Output: (flag, array of indices)
        flag for whether the needed value exists in the input tag  ---   bool
        segmented index of the input tag   ---   np.array
        """
        tag_array = np.array(tag)
        value_array = np.array(value)
        target_index = np.where(tag_array in value_array)[0]
        flag = True if len(target_index) > 0 else False
        split_index = np.where(np.diff(target_index) > 1)[
                          0] + 1  # split where the index of needed value is not continuous
        return flag, np.split(target_index, split_index)
