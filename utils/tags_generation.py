# imports and global setting

import numpy as np
from collections import namedtuple
from create_rect_from_file import get_parsed_data,get_agent_list,rect_object_creator
from shapely.ops import unary_union
from static_elements import StaticElementsWaymo
from long_activity_detector import long_act_detector
from lateral_activity_detector import lat_act_detector
import traceback
from logger.logger import *

# parameters
max_acc = [0,0.7,0.2,0.4,0]
a_cruise = [0,0.3,0.1,0.2,0]
delta_v = [0,1,0.2,0.5,0]
actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}
agent_state_dict = {"vehicle":{},"pedestrian":{},"cyclist":{}}
agent_pp_state_list = []

k_h=6
time_steps=91
# degree of smoothing spline
k=3
# default smoothing factor
smoothing_factor = time_steps
t_s = 0.1
kernel = 6
sampling_threshold = 8.72e-2  # 
time_steps = 91
integration_threshold = sampling_threshold*9 # 8.72e-2*9 = 0.785 rad. = 44.97 deg.
# parameter for estimation of the actor approaching a static element
TTC_1 = 5
# parameter for estimation of two actors' interaction
TTC_2 = 9
bbox_extension = 2 # extend length and width of the bbox by 2 times

def generate_tags(DATADIR,FILE:str):
    static_element = generate_lanes(DATADIR,FILE)
    lane_key = ['freeway','surface_street','bike_lane']
    other_object_key = ['cross_walk','speed_bump']
    controlled_lane = static_element.get_controlled_lane()
    actors_activity={} # [actor_type][actor_id][validity/lo_act/la_act]
    actors_static_element_relation = {} #[actor_type][actor_id][lane_type]
    actors_static_element_intersection = {} #[actor_type][actor_id][lane_type][expanded/trajectory],value is a list of area of intersection
    actors_list = {} #[actor_type]
    AgentExtendedPolygons = namedtuple('AgentExtendedPolygons','type,key,etp,ebb,length')
    agent_pp_state_list = []
    for actor_type in actor_dict:
        agent_type = actor_dict[actor_type]
        agent_list = get_agent_list(agent_type,DATADIR,FILE)
        ##############################################
        actor_activity = {}
        actor_static_element_intersection = {}
        if len(agent_list.shape) == 0:
            agent_list = [agent_list.item()]
            print(f"Processing {actor_type} with {len(agent_list)} agents...")
        else:
            print(f"Processing {actor_type} with {agent_list.shape[0]} agents...")
        agent_list_2 = agent_list.copy()
        for agent in agent_list:
            agent_activity = {}
            agent_static_element_intersection = {}
            agent_state,_ = rect_object_creator(agent_type,agent,DATADIR,FILE)
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
            agent_extended_polygons = AgentExtendedPolygons(actor_type,agent_key,etp,ebb,time_steps)
            agent_pp_state_list.append(agent_extended_polygons)
            lo_act,long_v,long_v1,knots = long_act_detector(agent_state,k_h,max_acc[agent_type],t_s=0.1,a_cruise=a_cruise[agent_type],\
                                            delta_v=delta_v[agent_type],time_steps=time_steps,k_cruise=10,\
                                            k=k,smoothing_factor=smoothing_factor)
            long_v = long_v.squeeze()
            lo_act = lo_act.squeeze()
            agent_activity['validity/appearance'] = validity_proportion
            agent_activity['lo_act'] = lo_act.tolist()
            agent_activity['long_v'] = long_v.tolist()
            la_act,bbox_yaw_rate = lat_act_detector(agent_state,t_s,sampling_threshold,integration_threshold,k=3,smoothing_factor=smoothing_factor)
            agent_activity['la_act'] = la_act.squeeze().tolist()
            agent_activity['yaw_rate'] = bbox_yaw_rate.squeeze().tolist()
            ###############################################
            agent_activity['valid'] = np.array([valid_start,valid_end],dtype=np.float32).tolist()
            actor_activity[f"{agent_key}_activity"] = agent_activity
            agent_state_dict[actor_type][agent_key] = agent_state
            # Generate actors in shapely
            actor_expanded_multipolygon = agent_state.expanded_polygon_set(TTC=TTC_1,sampling_fq=10)
            # actor_expanded_multipolygon = actor_expanded_polygon
            actor_trajectory_polygon = agent_state.polygon_set()
            # compute intersection with all lane types
            for lane_type in lane_key:
                agent_lane_intersection_expanded = np.zeros_like(lo_act).tolist()
                agent_lane_intersection_expanded_ratio = np.zeros_like(lo_act).tolist()
                agent_lane_intersection_trajectory = np.zeros_like(lo_act).tolist()
                agent_lane_intersection_trajectory_ratio = np.zeros_like(lo_act).tolist()
                lane_polygon_list = static_element.get_lane(lane_type)
                for step in range(valid_start,valid_end+1):
                    actor_expanded_multipolygon_step = actor_expanded_multipolygon[step]
                    actor_trajectory_polygon_step = actor_trajectory_polygon[step]
                    intersection,intersection_expanded = 0,0
                    
                    # intersection = unary_union(lane_polygon_list).intersection(actor_trajectory_polygon_step).area
                    # intersection_expanded = unary_union(lane_polygon_list).intersection(actor_expanded_multipolygon_step).area
                    
                    for lane_polygon in lane_polygon_list:
                        intersection_expanded += actor_expanded_multipolygon_step.intersection(lane_polygon).area
                        intersection += actor_trajectory_polygon_step.intersection(lane_polygon).area
                    agent_lane_intersection_expanded[step]=intersection_expanded
                    agent_lane_intersection_trajectory[step]=intersection
                    agent_lane_intersection_expanded_ratio[step] = intersection_expanded/actor_expanded_multipolygon_step.area
                    agent_lane_intersection_trajectory_ratio[step] = intersection/actor_trajectory_polygon_step.area
                agent_lane_relation = __compute_relation_actor_road_feature(valid_start,valid_end,agent_lane_intersection_trajectory_ratio,agent_lane_intersection_expanded_ratio)
                # for efficiency, we can only store the intersection area when any of the two ratios is greater than zero
                agent_static_element_intersection[lane_type]={
                    'relation':agent_lane_relation,
                    'expanded':agent_lane_intersection_expanded,
                    'expanded_ratio':agent_lane_intersection_expanded_ratio,
                    'trajectory':agent_lane_intersection_trajectory,
                    'trajectory_ratio':agent_lane_intersection_trajectory_ratio
                }
            # compute intersection with other types of objects
            for other_object_type in other_object_key:
                agent_other_object_intersection_expanded = np.zeros_like(lo_act).tolist()
                agent_other_object_intersection_trajectory = np.zeros_like(lo_act).tolist()
                agent_other_object_intersection_expanded_ratio = np.zeros_like(lo_act).tolist()
                agent_other_object_intersection_trajectory_ratio = np.zeros_like(lo_act).tolist()
                other_object_polygon_list = static_element.get_other_object(other_object_type)
                for step in range(valid_start,valid_end+1):
                    actor_expanded_multipolygon_step = actor_expanded_multipolygon[step]
                    actor_trajectory_polygon_step = actor_trajectory_polygon[step]
                    intersection,intersection_expanded = 0,0
                    # intersection = unary_union(other_object_polygon_list).intersection(actor_trajectory_polygon_step).area
                    # intersection_expanded = unary_union(other_object_polygon_list).intersection(actor_expanded_multipolygon_step).area
                    for other_object_polygon in other_object_polygon_list:
                        try:
                            intersection_expanded += actor_expanded_multipolygon_step.intersection(other_object_polygon).area
                            intersection += actor_trajectory_polygon_step.intersection(other_object_polygon).area
                        except Exception as e:
                            logger.error(f"FILE:{FILE},Intersection computation: {e}.\n type:{other_object_type},polygon:{other_object_polygon}")
                    agent_other_object_intersection_expanded[step]=intersection_expanded
                    agent_other_object_intersection_trajectory[step]=intersection
                    agent_other_object_intersection_expanded_ratio[step] = intersection_expanded/actor_expanded_multipolygon_step.area
                    agent_other_object_intersection_trajectory_ratio[step] = intersection/actor_trajectory_polygon_step.area
                agent_lane_relation = __compute_relation_actor_road_feature(valid_start,valid_end,agent_other_object_intersection_trajectory_ratio,agent_other_object_intersection_expanded_ratio)
                agent_static_element_intersection[other_object_type]={
                    'relation':agent_lane_relation,
                    'expanded':agent_other_object_intersection_expanded,
                    'expanded_ratio':agent_other_object_intersection_expanded_ratio,
                    'trajectory':agent_other_object_intersection_trajectory,
                    'trajectory_ratio':agent_other_object_intersection_trajectory_ratio
                }
            # compute intersection with controlled lanes
            controlled_lanes = static_element.get_controlled_lane()
            controlled_lanes_id = static_element.controlled_lanes_id
            traffic_lights_state = static_element.traffic_lights['traffic_lights_state']
            traffic_lights_id = static_element.traffic_lights['traffic_lights_lane_id']
            traffic_lights_points= static_element.traffic_lights['points']

            for controlled_lane,controlled_lane_id in zip(controlled_lanes,controlled_lanes_id):
                agent_controlled_lane_intersection_expanded = np.zeros_like(lo_act).tolist()
                agent_controlled_lane_intersection_trajectory = np.zeros_like(lo_act).tolist()
                agent_controlled_lane_intersection_expanded_ratio = np.zeros_like(lo_act).tolist()
                agent_controlled_lane_intersection_trajectory_ratio = np.zeros_like(lo_act).tolist()
                # intersection,intersection_expanded = [],[]
                # intersection_ratio,intersection_expanded_ratio = [],[]
                intersected_polygons,intersected_polygons_expanded = [],[]
                for step in range(valid_start,valid_end+1):
                    actor_expanded_multipolygon_step = actor_expanded_multipolygon[step]
                    actor_trajectory_polygon_step = actor_trajectory_polygon[step]
                    agent_controlled_lane_intersection_expanded[step]=actor_expanded_multipolygon_step.intersection(controlled_lane).area
                    agent_controlled_lane_intersection_trajectory[step]=actor_trajectory_polygon_step.intersection(controlled_lane).area
                    agent_controlled_lane_intersection_expanded_ratio[step] = actor_expanded_multipolygon_step.intersection(controlled_lane).area/actor_expanded_multipolygon_step.area
                    agent_controlled_lane_intersection_trajectory_ratio[step] = actor_trajectory_polygon_step.intersection(controlled_lane).area/actor_trajectory_polygon_step.area
                    intersected_polygons.append(actor_expanded_multipolygon_step.intersection(controlled_lane))
                    intersected_polygons_expanded.append(actor_expanded_multipolygon_step.intersection(controlled_lane))
                actor_lane_relation = __compute_relation_actor_road_feature(valid_start,valid_end,agent_controlled_lane_intersection_trajectory_ratio,agent_controlled_lane_intersection_expanded_ratio)
                if np.sum(agent_controlled_lane_intersection_expanded)==0 and np.sum(agent_controlled_lane_intersection_trajectory)==0:
                    flag_controlled_lane = False
                    continue
                else:
                    intersected_polygon = unary_union(intersected_polygons)
                    intersected_polygon_expanded = unary_union(intersected_polygons_expanded)
                    light_index = 0
                    for light_index_temp,traffic_light_point in enumerate(traffic_lights_points):
                        if np.sum(agent_controlled_lane_intersection_trajectory)>0 and intersected_polygon.contains(traffic_light_point):
                            light_index = light_index_temp
                            break
                        elif np.sum(agent_controlled_lane_intersection_expanded)>0 and intersected_polygon_expanded.contains(traffic_light_point):
                            light_index = light_index_temp
                            break
                    light_state = traffic_lights_state[:,light_index].tolist()
                    controlled_lane_key = f"controlled_lane_{controlled_lane_id}"
                    agent_static_element_intersection[controlled_lane_key]={
                        'relation':actor_lane_relation,
                        'light_state':light_state,
                        'expanded':agent_controlled_lane_intersection_expanded,
                        'expanded_ratio':agent_controlled_lane_intersection_expanded_ratio,
                        'trajectory':agent_controlled_lane_intersection_trajectory,
                        'trajectory_ratio':agent_controlled_lane_intersection_trajectory_ratio
                    }
            actor_static_element_intersection[agent_key] = agent_static_element_intersection
            road_graph_plot_flag=0
        if isinstance(agent_list_2,list):    
            actors_list[actor_type] = agent_list_2
        else:
            actors_list[actor_type] = agent_list_2.tolist()
        actors_activity[actor_type] = actor_activity
        actors_static_element_intersection[actor_type] = actor_static_element_intersection
    inter_actor_relation = __generate_inter_actor_relation(agent_pp_state_list)
    return actors_list,inter_actor_relation,actors_activity,actors_static_element_intersection

def generate_lanes(DATADIR:str,FILE:str):
    original_data_roadgragh,original_data_light = road_graph_parser(DATADIR,FILE)
    static_element = StaticElementsWaymo(original_data_roadgragh,original_data_light)
    static_element.create_polygon_set()
    return static_element

def road_graph_parser(DATADIR:str,FILE:str)->tuple:
    decoded_example = get_parsed_data(DATADIR,FILE)
    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()
    roadgraph_type = decoded_example['roadgraph_samples/type'].numpy()
    roadgraph_lane_id = decoded_example['roadgraph_samples/id'].numpy()
    roadgraph_dir_xyz = decoded_example['roadgraph_samples/dir'].numpy()
    # concatenate past,current and future states of traffic lights
    #[num_steps,num_light_positions]
    traffic_lights_id = np.concatenate([decoded_example['traffic_light_state/past/id'].numpy(),decoded_example['traffic_light_state/current/id'].numpy(),decoded_example['traffic_light_state/future/id'].numpy()],axis=0)
    traffic_lights_valid = np.concatenate([decoded_example['traffic_light_state/past/valid'].numpy(),decoded_example['traffic_light_state/current/valid'].numpy(),decoded_example['traffic_light_state/future/valid'].numpy()],axis=0)
    traffic_lights_state = np.concatenate([decoded_example['traffic_light_state/past/state'].numpy(),decoded_example['traffic_light_state/current/state'].numpy(),decoded_example['traffic_light_state/future/state'].numpy()],axis=0)
    traffic_lights_pos_x = np.concatenate([decoded_example['traffic_light_state/past/x'].numpy(),decoded_example['traffic_light_state/current/x'].numpy(),decoded_example['traffic_light_state/future/x'].numpy()],axis=0)
    traffic_lights_pos_y = np.concatenate([decoded_example['traffic_light_state/past/y'].numpy(),decoded_example['traffic_light_state/current/y'].numpy(),decoded_example['traffic_light_state/future/y'].numpy()],axis=0)
    original_data_roadgragh = {
        'roadgraph_xyz':roadgraph_xyz,
        'roadgraph_type':roadgraph_type,
        'roadgraph_dir_xyz':roadgraph_dir_xyz,
        'roadgraph_lane_id':roadgraph_lane_id
    }
    original_data_light = {
        'traffic_lights_id':traffic_lights_id,
        'traffic_lights_valid':traffic_lights_valid,
        'traffic_lights_state':traffic_lights_state,
        'traffic_lights_pos_x':traffic_lights_pos_x,
        'traffic_lights_pos_y':traffic_lights_pos_y
    }
    return original_data_roadgragh,original_data_light


def __compute_relation_actor_road_feature(valid_start,valid_end,trajectory_ratio,expanded_ratio):
    """
    compute the relation between the actor and the road features
    -5 ---  invalid
    -1 --- not relative
    0 --- leaving
    1 --- approaching
    2 --- entering
    3 --- staying
    """
    actor_lane_relation = np.ones_like(expanded_ratio)*(-1)
    trajectory_ratio = np.array(trajectory_ratio)
    expanded_ratio = np.array(expanded_ratio)
    interesting_threshold = 1e-2
    # ratio = 0 means not relative
    if np.sum(trajectory_ratio) == 0 and np.sum(expanded_ratio) == 0:
        # making sure the invalid is -5
        actor_lane_relation[:int(valid_start)] = -5
        actor_lane_relation[int(valid_end)+1:] = -5
        return actor_lane_relation.tolist()
    elif np.sum(trajectory_ratio) == 0 and np.sum(expanded_ratio) != 0:
        approaching = np.where(expanded_ratio>interesting_threshold)[0]
        actor_lane_relation[approaching] =1
        # making sure the invalid is -5
        actor_lane_relation[:int(valid_start)] = -5
        actor_lane_relation[int(valid_end)+1:] = -5
        return actor_lane_relation.tolist()
    else:
        # set the actual ratio greater than 1 to 1
        trajectory_ratio = np.where(trajectory_ratio>1,1,trajectory_ratio)
        difference_trajectory_ratio = np.diff(trajectory_ratio)
        relative_time = np.where(trajectory_ratio>interesting_threshold)[0]
        staying = np.intersect1d(np.where(np.abs(difference_trajectory_ratio)<=interesting_threshold)[0]+1,relative_time)
        entering = np.intersect1d(np.where(difference_trajectory_ratio>interesting_threshold)[0]+1,relative_time)
        leaving = np.intersect1d(np.where(difference_trajectory_ratio<-interesting_threshold)[0]+1,relative_time)
        actor_lane_relation[staying] = 3
        actor_lane_relation[entering] = 2
        actor_lane_relation[leaving] = 0
        actor_lane_relation[0] = actor_lane_relation[1]
        actor_lane_relation[valid_start]=actor_lane_relation[valid_start+1]
        # print(difference_trajectory_ratio,difference_trajectory_ratio.shape)
        # print(staying)
        if np.sum(expanded_ratio) != 0:
            approaching = np.where(expanded_ratio>0)[0]
            not_relative = np.where(trajectory_ratio<=interesting_threshold)[0]
            filtered_approaching = np.intersect1d(approaching,not_relative)
            # expanded_ratio > 0 and actor_lane_relation == -1 ==>approaching
            actor_lane_relation[filtered_approaching] =1
        # making sure the invalid is -5
        actor_lane_relation[:int(valid_start)] = -5
        actor_lane_relation[int(valid_end)+1:] = -5
        return actor_lane_relation.tolist()

def __generate_inter_actor_relation(agent_pp_state_list:list):
    """
    0 --- not related
    1 --- related by extended trajectory polygons   (etp or type 1)
    2 --- related by extended bounding boxes    (ebb or type 2)
    3 --- related by both etp and ebb (type 3)
    """
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
            relation = np.zeros(length)
            for step in range(length):
                # print(len(agent_etp_1[step]))
                # print(len(agent_etp_2[step]))
                # print(step)
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
                        relation[step] = 3
                    elif etp_flag and not intersection_ebb:
                        relation[step] = 1
                    elif not etp_flag and intersection_ebb:
                        relation[step] = 2
            if np.sum(relation):
                inter_actor_relation[agent_key_1][agent_key_2] = relation.tolist()
    return inter_actor_relation



