# imports and global setting
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from collections import namedtuple
from helpers.create_rect_from_file import get_agent_list,actor_creator
from shapely.ops import unary_union
from environ_elements import EnvironmentElementsWaymo
from long_act_detector import LongActDetector
from lateral_act_detector import LatActDetector
import traceback
from logger.logger import *
from parameters.tag_parameters import *
from parameters.tags_dict import *

class TagsGenerator:
    def __init__(self):
        self.tags = {
            'actors_list':[],
            'inter_actor_relation':[],
            'actors_activity':[],
            'actors_environment_element_intersection':[]
        }
    
    def __call__(self,DATADIR,FILE):
        return self.tagging(DATADIR,FILE)

    def __repr__(self) -> str:
        return f"TagsGenerator() tagging {self.tags.keys()}"

    def tagging(self,DATADIR,FILE:str):
        """
        tagging the actors in the scene
        """
        environment_element = self.generate_lane_polygons(DATADIR,FILE)

        agent_state_dict = {key:{} for key in actor_dict.keys()}
        agent_pp_state_list = []
        controlled_lane = environment_element.get_controlled_lane()
        actors_activity={} # [actor_type][actor_id][validity/lo_act/la_act]
        actors_environment_element_relation = {} #[actor_type][actor_id][lane_type]
        actors_environment_element_intersection = {} #[actor_type][actor_id][lane_type][expanded/trajectory],value is a list of area of intersection
        actors_list = {} #[actor_type] 
        AgentExtendedPolygons = namedtuple('AgentExtendedPolygons','type,key,etp,ebb,length,x,y,theta,v_dir')
        agent_pp_state_list = []
        for actor_type in actor_dict:
            agent_type = actor_dict[actor_type]
            agent_list = get_agent_list(agent_type,DATADIR,FILE)
            ##############################################
            actor_activity = {}
            actor_environment_element_intersection = {}
            if len(agent_list.shape) == 0:
                agent_list = [agent_list.item()]
                print(f"Tagging {len(agent_list)} {actor_type}...")
            else:
                print(f"Tagging {agent_list.shape[0]} {actor_type}s...")
            agent_list_2 = agent_list.copy()
            for agent in agent_list:
                # agent_activity = {}
                agent_environment_element_intersection = {}
                agent_state,_ = actor_creator(agent_type,agent,DATADIR,FILE)
                valid_start,valid_end = agent_state.get_validity_range()
                # smoothing factor equals to the number of valid time steps
                smoothing_factor = valid_end-valid_start + 1
                ###########################
                # not computing with only one step valid agent
                if valid_start == valid_end:
                    agent_idx = np.where(agent_list_2==agent)[0]
                    agent_list_2 = np.delete(agent_list_2,agent_idx)
                    continue
                validity_proportion = agent_state.data_preprocessing()
                agent_key = f"{actor_type}_{agent}"
                # extended trajectory pologons
                _ = agent_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=10)
                etp = agent_state.expanded_multipolygon
                # generate the extended bounding boxes
                ebb = agent_state.expanded_bbox_list(expand=bbox_extension)

                x = agent_state.kinematics['x']
                y = agent_state.kinematics['y']
                theta = agent_state.kinematics['bbox_yaw']
                v_dir = agent_state.kinematics['vel_yaw']
                #TODO: uniunivariate_spline v_dir

                agent_extended_polygons = AgentExtendedPolygons(actor_type,agent_key,etp,ebb,time_steps,x,y,theta,v_dir)
                agent_pp_state_list.append(agent_extended_polygons)
                ######### long activity detection ###########
                long_act_detector = LongActDetector()
                lo_act,long_v,long_v1,knots = long_act_detector.tagging(agent_state,k_h,max_acc[agent_type],t_s=0.1,a_cruise=a_cruise[agent_type],\
                                                delta_v=delta_v[agent_type],time_steps=time_steps,k_cruise=10,\
                                                k=k,smoothing_factor=smoothing_factor)
                long_v = long_v.squeeze().tolist()
                lo_act = lo_act.squeeze().tolist()
                agent_activity={
                    'validity/appearance': validity_proportion,
                    'lo_act': lo_act,
                    'long_v': long_v
                }
                #########  lateral activity detection #########
                lat_act_detector = LatActDetector()
                la_act,bbox_yaw_rate = lat_act_detector.tagging(agent_state,t_s,sampling_threshold,intgr_threshold_turn,intgr_threshold_swerv,k=3,smoothing_factor=smoothing_factor)
                agent_activity['la_act'] = la_act.tolist()
                agent_activity['yaw_rate'] = bbox_yaw_rate.tolist()
                ##########      other properties    ###########
                agent_activity['valid'] = np.array([valid_start,valid_end],dtype=np.float32).tolist()
                actor_activity[f"{agent_key}_activity"] = agent_activity
                agent_state_dict[actor_type][agent_key] = agent_state
                ######### interaction with environment elements #########
                # Generate actors polygon set in shapely
                actor_expanded_multipolygon = agent_state.expanded_polygon_set(TTC=TTC_1,sampling_fq=10)
                # actor_expanded_multipolygon = actor_expanded_polygon
                actor_trajectory_polygon = agent_state.polygon_set()
                # compute intersection with all lane types
                agent_lane_id = {key:[] for key in lane_key}
                for key in (lane_key+dashed_road_line_key):
                    expanded,expanded_ratio,\
                    traj,traj_ratio,\
                    current_controlled_lane,current_lane_id = self.__initialize_with_example(lo_act,6)

                    lane_polygon_list = environment_element.get_lane(key)
                    lane_id_list = environment_element.lane_id[key]
                    for step in range(valid_start,valid_end+1):
                        # actor_expanded_multipolygon[step] = actor_expanded_multipolygon[step]
                        # actor_trajectory_polygon[step] = actor_trajectory_polygon[step]
                        intersection,intersection_expanded = 0,0
                        # np.nan,int64 is not JSON serializable!
                        pos_lane_id = -99.0 # dummy lane id
                        pos_lane_intersection = 0
                        for (lane_polygon,lane_id) in zip(lane_polygon_list,lane_id_list):
                            intersection_expanded += actor_expanded_multipolygon[step].intersection(lane_polygon).area
                            current_actual_intersection = actor_trajectory_polygon[step].intersection(lane_polygon).area
                            if current_actual_intersection > pos_lane_intersection:
                                pos_lane_id = lane_id
                                pos_lane_intersection = current_actual_intersection
                            intersection += current_actual_intersection
                        
                        current_controlled_lane[step] = 1 if pos_lane_id in environment_element.controlled_lane_id else 0
                        expanded[step]=intersection_expanded
                        traj[step]=intersection
                        expanded_ratio[step] = intersection_expanded/actor_expanded_multipolygon[step].area
                        traj_ratio[step] = intersection/actor_trajectory_polygon[step].area
                        current_lane_id[step] = float(pos_lane_id)
                    
                    if key in dashed_road_line_key:
                        for lane_key_type in lane_key:
                            agent_lane_id[lane_key_type] =  agent_environment_element_intersection[lane_key_type]['current_lane_id']

                    agent_lane_relation = self.__compute_relation_actor_road_feature(valid_start,valid_end,traj_ratio,expanded_ratio,type=key,lane_id=agent_lane_id,la_act = agent_activity['la_act'])
                    # for efficiency, we can only store the intersection area when any of the two ratios is greater than zero
                    agent_environment_element_intersection[key]={
                        'relation':agent_lane_relation,
                        'expanded':expanded,
                        'expanded_ratio':expanded_ratio,
                        'trajectory':traj,
                        'trajectory_ratio':traj_ratio,
                        'current_lane_id':current_lane_id
                    }
                # compute intersection with other types of objects
                for other_object_type in other_object_key:
                    expanded,traj,expanded_ratio,traj_ratio = self.__initialize_with_example(lo_act,4)
                    other_object_polygon_list = environment_element.get_other_object(other_object_type)
                    for step in range(valid_start,valid_end+1):
                        # actor_expanded_multipolygon[step] = actor_expanded_multipolygon[step]
                        # actor_trajectory_polygon[step] = actor_trajectory_polygon[step]
                        intersection,intersection_expanded = 0,0
                        # intersection = unary_union(other_object_polygon_list).intersection(actor_trajectory_polygon[step]).area
                        # intersection_expanded = unary_union(other_object_polygon_list).intersection(actor_expanded_multipolygon[step]).area
                        for other_object_polygon in other_object_polygon_list:
                            try:
                                intersection_expanded += actor_expanded_multipolygon[step].intersection(other_object_polygon).area
                                intersection += actor_trajectory_polygon[step].intersection(other_object_polygon).area
                            except Exception as e:
                                logger.error(f"FILE:{FILE},Intersection computation: {e}.\n type:{other_object_type},polygon:{other_object_polygon}")
                        expanded[step]=intersection_expanded
                        traj[step]=intersection
                        expanded_ratio[step] = intersection_expanded/actor_expanded_multipolygon[step].area
                        traj_ratio[step] = intersection/actor_trajectory_polygon[step].area
                    agent_lane_relation = self.__compute_relation_actor_road_feature(valid_start,valid_end,traj_ratio,expanded_ratio)
                    agent_environment_element_intersection[other_object_type]={
                        'relation':agent_lane_relation,
                        'expanded':expanded,
                        'expanded_ratio':expanded_ratio,
                        'trajectory':traj,
                        'trajectory_ratio':traj_ratio
                    }
                # compute intersection with controlled lanes
                
                agent_controlled_lane = {}
                controlled_lanes = environment_element.get_controlled_lane()
                controlled_lanes_id = environment_element.controlled_lane_id
                traffic_lights_state = environment_element.traffic_lights['traffic_lights_state']
                traffic_lights_id = environment_element.traffic_lights['traffic_lights_lane_id']
                traffic_lights_points= environment_element.traffic_lights['points']

                for controlled_lane,controlled_lane_id in zip(controlled_lanes,controlled_lanes_id):
                    expanded,traj,expanded_ratio,traj_ratio = self.__initialize_with_example(lo_act,4)
                    intersected_polygons,intersected_polygons_expanded = [],[]
                    for step in range(valid_start,valid_end+1):
                        # actor_expanded_multipolygon[step] = actor_expanded_multipolygon[step]
                        # actor_trajectory_polygon[step] = actor_trajectory_polygon[step]
                        expanded[step]=actor_expanded_multipolygon[step].intersection(controlled_lane).area
                        traj[step]=actor_trajectory_polygon[step].intersection(controlled_lane).area
                        expanded_ratio[step] = actor_expanded_multipolygon[step].intersection(controlled_lane).area/actor_expanded_multipolygon[step].area
                        traj_ratio[step] = actor_trajectory_polygon[step].intersection(controlled_lane).area/actor_trajectory_polygon[step].area
                        intersected_polygons.append(actor_expanded_multipolygon[step].intersection(controlled_lane))
                        intersected_polygons_expanded.append(actor_expanded_multipolygon[step].intersection(controlled_lane))
                    actor_lane_relation = self.__compute_relation_actor_road_feature(valid_start,valid_end,traj_ratio,expanded_ratio)
                    if np.sum(expanded)==0 and np.sum(traj)==0:
                        flag_controlled_lane = False
                        continue
                    else:
                        intersected_polygon = unary_union(intersected_polygons)
                        intersected_polygon_expanded = unary_union(intersected_polygons_expanded)
                        light_index = 0
                        for light_index_temp,traffic_light_point in enumerate(traffic_lights_points):
                            if np.sum(traj)>0 and intersected_polygon.contains(traffic_light_point):
                                light_index = light_index_temp
                                break
                            elif np.sum(expanded)>0 and intersected_polygon_expanded.contains(traffic_light_point):
                                light_index = light_index_temp
                                break
                        light_state = traffic_lights_state[:,light_index].tolist()
                        controlled_lane_key = f"controlled_lane_{controlled_lane_id}"
                        agent_controlled_lane[controlled_lane_key]={
                            'relation':actor_lane_relation,
                            'light_state':light_state,
                            'expanded':expanded,
                            'expanded_ratio':expanded_ratio,
                            'trajectory':traj,
                            'trajectory_ratio':traj_ratio
                        }
                # TODO: fix the current controlled lane with approaching
                ctrl_lane_id = self.__generate_agent_ctrl_lane_relation(agent_controlled_lane)
                if "_" in ctrl_lane_id or ctrl_lane_id != 'None':
                    agent_environment_element_intersection["controlled_lane"] = {
                        "lane_id" : ctrl_lane_id.split("_")[-1],
                        "relation" : agent_controlled_lane[ctrl_lane_id]['relation'],
                        "light_state" : agent_controlled_lane[ctrl_lane_id]['light_state'],
                        "expanded" : agent_controlled_lane[ctrl_lane_id]['expanded'],
                        "expanded_ratio" : agent_controlled_lane[ctrl_lane_id]['expanded_ratio'],
                        "trajectory" : agent_controlled_lane[ctrl_lane_id]['trajectory'],
                        "trajectory_ratio" : agent_controlled_lane[ctrl_lane_id]['trajectory_ratio']
                    }
                actor_environment_element_intersection[agent_key] = agent_environment_element_intersection
                road_graph_plot_flag=0
            if isinstance(agent_list_2,list):    
                actors_list[actor_type] = agent_list_2
            else:
                actors_list[actor_type] = agent_list_2.tolist()
            actors_activity[actor_type] = actor_activity
            actors_environment_element_intersection[actor_type] = actor_environment_element_intersection
        ########### inter actor relation ###########
        inter_actor_relation = self.__generate_inter_actor_relation(agent_pp_state_list)
        ########### general info ###########
        general_info = {
            'actors_list': actors_list,
            'tagging_parameters': tags_param,
        }
        return general_info,inter_actor_relation,actors_activity,actors_environment_element_intersection
    
    def __generate_agent_ctrl_lane_relation(self,agent_controlled_lane:dict)->str:
        """
        input: traj_ratio,expanded_ratio
        output: dict:{relation with one ctrl. lane, light_state}
        """
        ctrl_lane_id,traj_r = 'None',0
        for key in agent_controlled_lane.keys():
            temp_traj_r = np.sum(np.array(agent_controlled_lane[key]['trajectory_ratio'],dtype=np.float16))
            if temp_traj_r>traj_r:
                ctrl_lane_id = key
                traj_r = temp_traj_r
        return ctrl_lane_id

    def generate_lane_polygons(self,DATADIR:str,FILE:str):
        environment_element_waymo = EnvironmentElementsWaymo(DATADIR,FILE)
        environment_element_waymo.create_polygon_set()
        return environment_element_waymo
    
    def __initialize_with_example(self,example,number)->list:
        initialized = [np.zeros_like(example).tolist() for _ in range(number)]
        return initialized

    def __compute_intersection(self,lo_act,element_key1,actor_trajectory_polygon,actor_expanded_multipolygon,valid_start,valid_end,agent_environment_element_intersection:dict,element_key2=None):
        """
        compute the intersection between actor and environment element
        TODO: transfer the intersection funtion from tagging() to here
        """
        if element_key2 is not None:
            element_key = element_key1+element_key2
        else:
            element_key = element_key1
        for element_type in element_key:
            agent_element_intersection_expanded = np.zeros_like(lo_act).tolist()
            agent_element_intersection_expanded_ratio = np.zeros_like(lo_act).tolist()
            agent_element_intersection_trajectory = np.zeros_like(lo_act).tolist()
            agent_element_intersection_trajectory_ratio = np.zeros_like(lo_act).tolist()
            agent_current_element_id = np.zeros_like(lo_act).tolist()
        pass

    def __compute_relation_actor_road_feature(self,valid_start,valid_end,trajectory_ratio,expanded_ratio,type=None,lane_id=None,la_act=None):
        """
        compute the relation between the actor and the road features
        refer the parameters.tag_dict for the meaning of the relation 
        """
        new_tag_dict = dict([val,key] for key,val in road_relation_dict.items())

        if type in dashed_road_line_key:
            return self.__compute_actor_road_lane_change(valid_start,valid_end,trajectory_ratio,lane_id,la_act)
        actor_lane_relation = np.ones_like(expanded_ratio) * float(new_tag_dict["not relative"])
        trajectory_ratio = np.array(trajectory_ratio)
        expanded_ratio = np.array(expanded_ratio)

        interesting_threshold = 1e-2
        # ratio <= interesting threshold is not relative
        if np.sum(trajectory_ratio) == 0 and np.sum(expanded_ratio) == 0:
            pass
        elif np.sum(trajectory_ratio) == 0 and np.sum(expanded_ratio) != 0:
            approaching = np.where(expanded_ratio>interesting_threshold)[0]
            actor_lane_relation[approaching] = float(new_tag_dict["approaching"])
        else:
            # set the actual ratio greater than 1 to 1
            trajectory_ratio = np.where(trajectory_ratio>1,1,trajectory_ratio)
            trajectory_ratio_dot = np.diff(trajectory_ratio)
            trajectory_ratio_dot_dot = np.diff(trajectory_ratio_dot)
            relative_time = np.where(trajectory_ratio>interesting_threshold)[0]
            staying = np.intersect1d(np.where(np.abs(trajectory_ratio_dot)<=interesting_threshold)[0]+1,relative_time)
            entering = np.intersect1d(np.where(trajectory_ratio_dot>interesting_threshold)[0]+1,relative_time)
            leaving = np.intersect1d(np.where(trajectory_ratio_dot<-interesting_threshold)[0]+1,relative_time)
            actor_lane_relation[staying] = float(new_tag_dict["staying"])
            actor_lane_relation[entering] = float(new_tag_dict["entering"])
            actor_lane_relation[leaving] = float(new_tag_dict["leaving"])
            actor_lane_relation[0] = actor_lane_relation[1]
            actor_lane_relation[valid_start]=actor_lane_relation[valid_start+1]
            if np.sum(expanded_ratio) != 0:
                approaching = np.where(expanded_ratio>0)[0]
                not_relative = np.where(trajectory_ratio<=interesting_threshold)[0]
                filtered_approaching = np.intersect1d(approaching,not_relative)
                # expanded_ratio > 0 and actor_lane_relation == -1 ==>approaching
                actor_lane_relation[filtered_approaching] = float(new_tag_dict["approaching"])
        # making sure the invalid is -5
        actor_lane_relation[:int(valid_start)] = float(new_tag_dict["invalid"])
        actor_lane_relation[int(valid_end)+1:] = float(new_tag_dict["invalid"])
        return actor_lane_relation.tolist()

    def __compute_actor_road_lane_change(self,valid_start,valid_end,trajectory_ratio,lane_id,la_act): 
        """
        compute time instances when lane change happens
        refer the parameters.tag_dict for the meaning of the relation 
        """
        new_tag_dict = dict([val,key] for key,val in road_relation_dict.items())

        actor_road_lane_change = np.ones_like(trajectory_ratio) * float(new_tag_dict["no lane change"])
        trajectory_ratio = np.array(trajectory_ratio)
        none_zero_segments = np.split(np.where(trajectory_ratio!=0)[0],np.where(np.diff(np.nonzero(trajectory_ratio)[0])>1)[0]+1)
        abs_la_act = np.abs(la_act)

        for key in lane_key:
            for segment in none_zero_segments:
                if len(segment)>1:
                    lane_id_segment = lane_id[key][segment[0]:segment[-1]+1]
                    abs_la_act_segment = abs_la_act[segment[0]:segment[-1]+1]
                    if len(np.where(np.diff(lane_id_segment)>0)[0]) and len(np.where(abs_la_act_segment==2)[0]):
                        actor_road_lane_change[segment[0]:segment[-1]+1] = float(new_tag_dict["lane changing"])
        actor_road_lane_change[:int(valid_start)] = float(new_tag_dict["invalid"])
        actor_road_lane_change[int(valid_end)+1:] = float(new_tag_dict["invalid"])

        return actor_road_lane_change.tolist()

    def __generate_inter_actor_relation(self,agent_pp_state_list:list):
        """
        generate the inter actor relation
        refer the parameters.tag_dict for the meaning of the relation 
        """
        print(f"generating inter actor relation...")
        new_tag_dict = dict([val,key] for key,val in inter_actor_relation_dict.items())
        inter_actor_relation = {}
        for agent_pp_state_1 in agent_pp_state_list:
            agent_key_1 = agent_pp_state_1.key
            # agent_type_1 = agent_pp_state_1.type
            agent_etp_1 = agent_pp_state_1.etp
            agent_ebb_1 = agent_pp_state_1.ebb
            length = agent_pp_state_1.length
            inter_actor_relation[agent_key_1] = {}
            for agent_pp_state_2 in agent_pp_state_list:
                if agent_pp_state_2.key == agent_key_1:
                    continue
                else:
                    agent_key_2 = agent_pp_state_2.key
                # agent_type_2 = agent_pp_state_2.type
                agent_etp_2 = agent_pp_state_2.etp
                agent_ebb_2 = agent_pp_state_2.ebb
                relation = np.ones(length) * float(new_tag_dict['not related'])
                position = np.ones(length) * float(new_tag_dict['not related'])
                vel_dir = np.ones(length) * float(new_tag_dict['not related'])
                for step in range(length):
                    if agent_etp_1[step][0].area == 0 or agent_etp_2[step][0].area == 0:
                        continue
                    else:
                        etp_flag = 0
                        for i,(polygon_1,polygon_2) in enumerate(zip(agent_etp_1[step],agent_etp_2[step])):
                            intersection_etp = polygon_1.intersection(agent_etp_2[step][i]).area
                            if intersection_etp:
                                etp_flag = 1
                                break
                        intersection_ebb = agent_ebb_1[step].intersection(agent_ebb_2[step]).area
                        if etp_flag and intersection_ebb:
                            relation[step] = float(new_tag_dict['estimated collision+close proximity'])
                        elif etp_flag and not intersection_ebb:
                            relation[step] = float(new_tag_dict['estimated collision'])
                        elif not etp_flag and intersection_ebb:
                            relation[step] = float(new_tag_dict['close proximity'])
                        position[step] = self.__compute_actor_position_relation(agent_pp_state_1,agent_pp_state_2,step)
                        try:
                            vel_dir[step] = self.__compute_inter_actor_v_dir(agent_pp_state_1,agent_pp_state_2,step)
                        except:
                            raise ValueError(f"Agent1: {agent_key_1}, Agent2: {agent_key_2}, Step: {step}.")
                        #compute the position relation
                if np.sum(relation):
                    inter_actor_relation[agent_key_1][agent_key_2] = {}
                    inter_actor_relation[agent_key_1][agent_key_2]['relation'] = relation.tolist()
                    inter_actor_relation[agent_key_1][agent_key_2]['position'] = position.tolist()
                    inter_actor_relation[agent_key_1][agent_key_2]['v_dir'] = vel_dir.tolist()

        return inter_actor_relation

    def __compute_actor_position_relation(self,agent_pp_state_1,agent_pp_state_2,step):
        """
        compute the position relation between two agents
        refer the parameters.tag_dict for the meaning of the relation 
        """
        new_tag_dict = dict([val,key] for key,val in inter_actor_position_dict.items())
        theta = agent_pp_state_1.theta[step]
        position_relation_vector = np.array([agent_pp_state_2.x[step]-agent_pp_state_1.x[step],agent_pp_state_2.y[step]-agent_pp_state_1.y[step]])
        heading_vector = np.array([np.cos(theta),np.sin(theta)])
        cos_ = np.dot(position_relation_vector,heading_vector)/(np.linalg.norm(position_relation_vector))
        sin_ = np.cross(position_relation_vector,heading_vector)/(np.linalg.norm(position_relation_vector))

        # np.arctan2 (-pi,pi)
        if -0.25 * np.pi < np.arctan2(sin_,cos_) <= 0.25 * np.pi:
            position_relation = new_tag_dict['front']   # front
        elif 0.25 * np.pi < np.arctan2(sin_,cos_) <= 0.75 * np.pi:
            position_relation = new_tag_dict['left']    # left
        elif -0.75 * np.pi < np.arctan2(sin_,cos_) <= -0.25 * np.pi:
            position_relation = new_tag_dict['right']   # right
        else:
            position_relation = new_tag_dict['back']    # back
        return position_relation

    def __compute_inter_actor_v_dir(self,agent_pp_state_1,agent_pp_state_2,step):
        """
        compute the velocity direction relation between two agents
        refer the parameters.tag_dict for the meaning of the relation 
        """
        new_tag_dict = dict([val,key] for key,val in inter_actor_vel_dir_dict.items())
        v_relative_dir = np.array(agent_pp_state_2.v_dir[step]-agent_pp_state_1.v_dir[step])
        if -0.25 * np.pi < v_relative_dir <= 0.25 * np.pi:
            vel_dir_relation = new_tag_dict['same']
        elif 0.25 * np.pi < v_relative_dir <= 0.75 * np.pi:
            vel_dir_relation = new_tag_dict['left']
        elif -0.75 * np.pi < v_relative_dir <= -0.25 * np.pi:
            vel_dir_relation = new_tag_dict['right']
        else:
            vel_dir_relation = new_tag_dict['opposite']
        return vel_dir_relation
    



