import numpy as np

from parameters.tags_dict import *


class ScenarioMiner:
    def __init__(self) -> None:
        pass

    def mining(self, tags_dict: dict) -> dict:
        actors_list = tags_dict['general_info']['actors_list']
        inter_actor_relation = tags_dict['inter_actor_relation']
        actors_activity = tags_dict['actors_activity']
        actors_environment_element_intersection = tags_dict['actors_environment_element_intersection']
        solo_scenarios = {}
        for (actor_type, actors) in actors_list.items():
            # if not isinstance(actors,Iterable):
            #     actors = [actors]
            solo_scenarios[actor_type] = {}
            print(f"Mining solo scenario {len(actors)} {actor_type}(s).")

            for agent in actors:
                agent_key = f"{actor_type}_{agent}"
                solo_scenarios[actor_type][agent_key] = {}
                ##### longitudinal and lateral activity #####
                agent_activity = actors_activity[actor_type][f"{agent_key}_activity"]
                (valid_start, valid_end) = agent_activity['valid']
                valid_start = int(valid_start)
                valid_end = int(valid_end)
                agent_lo_act = agent_activity['lo_act']
                turning_points = self.__computing_turning_point(agent_lo_act, valid_start, valid_end)
                solo_scenarios[actor_type][agent_key]['lo_act'] = self.__summarizing_events(agent_lo_act,
                                                                                            turning_points, valid_start,
                                                                                            valid_end, 'lo',
                                                                                            lo_act_dict)
                agent_la_act = agent_activity['la_act']
                turning_points = self.__computing_turning_point(agent_la_act, int(valid_start), int(valid_end))
                solo_scenarios[actor_type][agent_key]['la_act'] = self.__summarizing_events(agent_la_act,
                                                                                            turning_points, valid_start,
                                                                                            valid_end, 'la',
                                                                                            la_act_dict)
                ##### agent and environment element relation #####
                agent_elements_relation = actors_environment_element_intersection[actor_type][agent_key]
                solo_scenarios[actor_type][agent_key]['environment'] = {}
                for (element_type, element) in agent_elements_relation.items():
                    relation = element['relation']
                    if sum(relation[valid_start:valid_end + 1]) == -(valid_end - valid_start + 1):
                        continue
                    else:
                        turning_points = self.__computing_turning_point(relation, valid_start, valid_end)
                        solo_scenarios[actor_type][agent_key]['environment'][element_type] = self.__summarizing_events(
                            relation, turning_points, valid_start, valid_end, element_type, road_relation_dict)
                        if element_type.startswith("controlled_lane"):
                            turning_points = self.__computing_turning_point(element['light_state'], valid_start,
                                                                            valid_end)
                            solo_scenarios[actor_type][agent_key]['environment'][element_type][
                                'light_state'] = self.__summarizing_events(element['light_state'], turning_points,
                                                                           valid_start, valid_end, 'light',
                                                                           light_state_dict)
                ##### inter-actor relation #####
                inter_actor = inter_actor_relation[agent_key]
                solo_scenarios[actor_type][agent_key]['inter_actor'] = {}
                for (actor_name, inter_actor_relation_position) in inter_actor.items():
                    solo_scenarios[actor_type][agent_key]['inter_actor'][actor_name] = {}
                    for (handel, value) in inter_actor_relation_position.items():
                        turning_points = self.__computing_turning_point(value, valid_start, valid_end)
                        if handel == 'position':
                            inter_actor_dict = inter_actor_position_dict
                        elif handel == 'relation':
                            inter_actor_dict = inter_actor_relation_dict
                        else:
                            inter_actor_dict = inter_actor_heading_dict
                        try:
                            solo_scenarios[actor_type][agent_key]['inter_actor'][actor_name][
                                handel] = self.__summarizing_events(value, turning_points, valid_start, valid_end,
                                                                    actor_name, inter_actor_dict)
                        except Exception as e:
                            raise ValueError(f"Error: {e}.\n{actor_name} {handel} {value}")
        return solo_scenarios

    def __computing_turning_point(self, activity, valid_start: int, valid_end: int) -> list:
        """
        find turning points of activity.
        i.e. the start of another type of activity
        """
        activity_diff = np.diff(np.array(activity[valid_start:valid_end + 1]))
        turning_points = np.where(activity_diff != 0)[0] + valid_start + 1
        return np.insert(turning_points, 0, valid_start).tolist()
        # turning_points = [valid_start]
        # for i in range(valid_start+1,valid_end):
        #     if activity[i] != activity[i-1]:
        #         turning_points.append(i)
        # return turning_points

    def __summarizing_events(self, activity, turning_points: list, valid_start: int, valid_end: int, event_type: str,
                             activity_type: dict) -> dict:
        """
        summarize events from activity and turning point
        event_type: tags_dict.keys()
        format:
        key:{event_type}_{number}
        value:{activity_type,start_frame,end_frame}
        """
        events_summary = {}
        for i, turning_point in enumerate(turning_points):
            event = activity_type[str(int(activity[turning_point]))]
            events_summary[f'{event_type}_{i}'] = {
                'event': event,
                'start': turning_point,
                'end': turning_points[i + 1] if i + 1 < len(turning_points) else valid_end
            }
        return events_summary
