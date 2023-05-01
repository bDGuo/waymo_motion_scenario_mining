import json
from typing import Dict
import numpy as np
import pandas as pd
import sys

from parameters.tag_parameters import sampling_frequency
from parameters.tags_dict import ReTagDict


class Counter:

    def __init__(self, result: dict, eval_mode=False):
        self.result = result
        self.time = ['0-3', '3-6', '6-e']
        self.act = ['lo_act', 'la_act']
        self.environment = ['surface_street', 'cross_walk', 'bike_lane']
        self.interactor = ['relation', 'position', 'heading']
        self.sc = ["SC1", "SC7", "SC13"]

    def count_tag(self, tag: str):
        count = self.__init_counter_tag(tag)
        for actor_type in self.result.keys():
            type_tag = self.result.get(actor_type, {})
            for actor in type_tag.keys():
                actor_tag = type_tag.get(actor, {})
                if tag in self.act:
                    count = self.__act_count(actor_tag.get(tag, {}), actor_type, count)
                if tag in self.environment:
                    envr_tag = actor_tag.get('environment', {}).get(tag, {})
                    count = self.__enr_count(envr_tag, actor_type, count)
                if tag in self.interactor:
                    # only count V2V, V2P, V2C
                    if actor_type != 'vehicle':
                        continue
                    interactor_tag = actor_tag.get('inter_actor', {})
                    count = self.__interactor_count(tag, interactor_tag, count)
        return count

    def count_sc(self):
        count = self.__init_counter_sc()
        for i in self.result.keys():
            sc = self.result.get(i, {})
            sc_id = sc.get("SC_ID", None)
            duration = len(sc.get('time_stamp', [])) / sampling_frequency
            if duration < 3:
                count.loc['0-3', sc_id] += 1
            elif duration < 6:
                count.loc['3-6', sc_id] += 1
            else:
                count.loc['6-e', sc_id] += 1
        return count

    def __interactor_count(self, tag: str, interactor: dict, count: pd.DataFrame):
        if not interactor:
            return count
        for interactor_k, interactor_v in interactor.items():
            actor_type = interactor_k.split('_')[0]
            guest_actor_tag = interactor_v.get(tag, {})
            for inter_tag_k, inter_tag_v in guest_actor_tag.items():
                duration = (inter_tag_v.get('end', 0) - inter_tag_v.get('start', 0)) / sampling_frequency
                event = inter_tag_v.get('event', None)
                if duration < 3:
                    count.loc[(event, '0-3'), actor_type] += 1
                elif duration < 6:
                    count.loc[(event, '3-6'), actor_type] += 1
                else:
                    count.loc[(event, '6-e'), actor_type] += 1
        return count

    def __enr_count(self, enr: dict, actor_type: str, count: pd.DataFrame):
        if not enr:
            return count
        for envr in enr:
            duration = (enr[envr].get('end', 0) - enr[envr].get('start', 0)) / sampling_frequency
            event = enr[envr].get('event', None)
            if duration < 3:
                count.loc[(event, '0-3'), actor_type] += 1
            elif duration < 6:
                count.loc[(event, '3-6'), actor_type] += 1
            else:
                count.loc[(event, '6-e'), actor_type] += 1
        return count

    def __act_count(self, act: dict, actor_type: str, count: pd.DataFrame):
        if not act:
            return count
        for activity in act:
            duration = (act[activity].get('end', 0) - act[activity].get('start', 0)) / sampling_frequency
            event = act[activity].get('event', None)
            if duration < 3:
                count.loc[(event, '0-3'), actor_type] += 1
            elif duration < 6:
                count.loc[(event, '3-6'), actor_type] += 1
            else:
                count.loc[(event, '6-e'), actor_type] += 1
        return count

    def __init_counter_sc(self) -> pd.DataFrame:
        counter = pd.DataFrame(np.zeros((len(self.time), 4)),
                               columns=["SC1", "SC7", "SC13", "time"])
        counter.loc[:, "time"] = self.time
        return counter.set_index(["time"])

    def __init_counter_tag(self, tag: str):
        if tag in self.act:
            tag_name = ReTagDict[tag].keys()
        if tag in self.environment:
            tag_name = ReTagDict['road_relation'].keys()
        if tag in self.interactor:
            tag_name = ReTagDict[f'inter_actor_{tag}'].keys()
        event = list(tag_name)
        time = self.time
        counter = pd.DataFrame(np.zeros((len(time) * len(event), 6)),
                               columns=['vehicle', 'pedestrian', 'cyclist', 'event', 'time', 'tag'])
        for i in range(len(event) * len(time)):
            counter.loc[i, 'event'] = event[i // len(time)]
            counter.loc[i, 'time'] = time[i % len(time)]
        counter.loc[:, 'tag'] = tag

        return counter.set_index(['event', 'time'])
