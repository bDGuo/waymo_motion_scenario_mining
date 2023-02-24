import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Union, Iterable, Any
from collections import namedtuple
from parameters.tags_dict import *
from parameters.scenario_categories import ScenarioCatelog

class ScenarioCategorizer:
    """
    Categorizes the scenarios into 11 categories.
    TODO: finish test on SC1 before moving on. Due date: 2023-02-24
    """
    def __init__(self, FILENUM:str, result_dict:dict):
        self.FILENUM = FILENUM
        self.actors_list = result_dict['general_info']['actors_list']
        self.inter_actor_relation = result_dict['inter_actor_relation']
        self.actors_activity = result_dict['actors_activity']
        self.actors_environment_element_intersection = result_dict['actors_environment_element_intersection']
        self.scenario_catelog = ScenarioCatelog

    def find_SC(self,scenario_category_ID:str):
        """
        Input: Scenario category ID, e.g. SC1
        Output: actor ID, start time and end time
        TODO: start with SC1
        """
        SC = self.scenario_catelog[scenario_category_ID]
        SC_result = {}
        for host_actor_type in SC.host_actor_type:
            for host_actor_id in self.actors_list[host_actor_type]:
                #####   encode host lo_act    #####
                flag_lo,lo_tag_encoded_h = self._tag_encoder(SC,self.actors_activity[host_actor_type][f'{host_actor_type}_{host_actor_id}_activity']['lo_act'],'lo_act')
                #####   encode host la_act    #####
                flag_la,la_tag_encoded_h = self._tag_encoder(SC,self.actors_activity[host_actor_type][f'{host_actor_type}_{host_actor_id}_activity']['la_act'],'la_act')
                time_stamp = lo_tag_encoded_h * la_tag_encoded_h
                #  early jump to the next host actor
                if not np.any(np.where(time_stamp==1)[0]):
                    continue
                #####   encode host road_relation   #####
                # TODO: No need for SC1,SC5,SC8
                #####   check guest actor   #####
                for guest_actor in self.inter_actor_relation[f'{host_actor_type}_{host_actor_id}'].keys():
                    #####   encode inter_actor_relation and position  #####                    
                    flag_relation,relation_tag_encoded_h = self._tag_encoder(SC,
                                                                            self.inter_actor_relation[f'{host_actor_type}_{host_actor_id}'][guest_actor]['relation']
                                                                            ,'inter_actor_relation')
                    # TODO: position is different. For now, it is just left,right, and front.
                    # should be left or right, then front.
                    flag_position,position_tag_encoded_h = self._tag_encoder(SC,
                                                                            self.inter_actor_relation[f'{host_actor_type}_{host_actor_id}'][guest_actor]['position']
                                                                            ,'inter_actor_position')
                    time_stamp *= relation_tag_encoded_h * position_tag_encoded_h
                    #   early jump to the next guest actor
                    if not np.any(np.where(time_stamp==1)[0]):
                        continue
                    #####    guest_actor   #####
                    guest_actor_type,guest_actor_id = guest_actor.split('_')
                    #####   check guest_actor type    #####
                    if not guest_actor_type in SC.guest_actor['actor_type']:
                        continue
                    #####   encode guest_actor lo_act    #####
                    flag_lo,lo_tag_encoded_g = self._tag_encoder(SC,self.actors_activity[guest_actor_type][f'{guest_actor}_activity']['lo_act'],'lo_act',host=False)
                    #####   encode guest_actor la_act    #####
                    flag_la,la_tag_encoded_g = self._tag_encoder(SC,self.actors_activity[guest_actor_type][f'{guest_actor}_activity']['la_act'],'la_act',host=False)
                    time_stamp *= lo_tag_encoded_g * la_tag_encoded_g
                    #  early jump to the next guest actor
                    if not np.any(np.where(time_stamp==1)[0]):
                        continue
                    #####   encode guest_actor road_relation #####
                    # TODO: No need for SC1
                    #####   result #####
                    SC_result = {
                        'SC_ID':scenario_category_ID,
                        'host_actor':host_actor_id,
                        'guest_actor':guest_actor_id,
                        'time_stamp':time_stamp.tolist()
                    }
        return SC_result
    
    def _tag_encoder(self,SC,encoding_tag:List,tag_type:str,host:bool=True):
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
        tag_type : the tag type required in the scenario category
        host: whether the actor is host or guest
        ----------------
        Output: (flag, encoded tag)
        """
        actor_constraint = SC.host_actor_tag if host else SC.guest_actor_tag
        tag = actor_constraint[tag_type]
        value = [float(k) for (k,v) in TagDict[tag_type].items() if v in tag]
        tag_array = np.array(encoding_tag)
        value_array = np.array(value)
        encoded_tag = np.zeros_like(tag_array)
        encoding_index = np.array([])
        for v in value_array:
            encoding_index = np.append(encoding_index,np.where(tag_array==v)[0]).astype(int)
        flag = np.any(encoding_index)
        encoded_tag[encoding_index] = 1
        return flag, encoded_tag
    
    def _tag_segmentation(self,tag:list,value:list):
        """
        Segmenting of the tags according to the needed value(s)
        ----------------
        Output: (flag, array of indices)
        flag for whether the needed value exists in the input tag  ---   bool
        segemented index of the input tag   ---   np.darray
        """
        tag_array = np.array(tag)
        value_array = np.array(value)
        target_index = np.where(tag_array in value_array)[0]
        flag = True if len(target_index)>0 else False
        split_index = np.where(np.diff(target_index)>1)[0]+1 # split where the index of needed value is not continuous
        return flag, np.split(target_index,split_index)