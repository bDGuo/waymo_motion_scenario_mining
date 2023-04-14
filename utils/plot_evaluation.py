"""
generate figures for evaluation using CARLA data
Author:Detian Guo
Date: 04/11/2022
"""
from collections import namedtuple, OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from rich.progress import track
import json
import argparse
from plotting_scenarios import get_color_map,plot_road_graph
from environ_elements import EnvironmentElementsWaymo
from parameters.plot_parameters import *
from parameters.tag_parameters import TTC_2, sampling_frequency,bbox_extension
from helpers.create_rect_from_file import get_parsed_data, get_parsed_carla_data, actor_creator
import re
from parameters.tags_dict import TagDict

# working directory
ROOT = Path(__file__).parent.parent

def plot_single_image(parsed,FILE,actors_states_list:list,i:int, time_steps:int, s_e, o_d_r, o_d_l,save_path,eval_mode=False):
    plt.rc('font',family='Times New Roman',size=font1['size'])
    fig,ax = plt.subplots(1,1,figsize=(15,15),dpi=100)
    ax,_,_,_=plot_road_graph(parsed,ax,s_e, o_d_r, o_d_l,eval_mode=eval_mode)
    ax_title = f"Time step: {i} of {time_steps}\n"
    for actor_state in actors_states_list:
        agent_type = actor_state.actor_type
        agent = actor_state.agent
        agent_state = actor_state.agent_state
        v_s,v_e = actor_state.v_s, actor_state.v_e
        tp = actor_state.tp
        etp = actor_state.etp
        ebb = actor_state.ebb
        ax = plot_actor_polygon(etp,i,v_s,v_e,ax,f"{agent_type}-{agent}-ETP",f"{agent_type}_etp",gradient=True)
        ax = plot_actor_polygon(ebb,i,v_s,v_e,ax,f"{agent_type}-{agent}-EBB",f"{agent_type}_ebb",gradient=False)
        ax = plot_actor_polygon(tp,i,v_s,v_e,ax,f"{agent_type}-{agent}-TP",f"{agent_type}_tp",gradient=False)
        lo_act = str(actor_state.agent_tag['lo_act'][i]).split(".")[0]
        la_act = str(actor_state.agent_tag['la_act'][i]).split(".")[0]
        surface_street = str(actor_state.agent_tag['environment']['surface_street'][i]).split(".")[0]
        cross_walk = str(actor_state.agent_tag['environment']['cross_walk'][i]).split(".")[0]
        ax_title = f"{ax_title}\
            {agent_type}_{agent}: lo_act:{TagDict['lo_act'][lo_act]}\nla_act:{TagDict['la_act'][la_act]}\n"
        ax_title = f"{ax_title}\
            surface_street: {TagDict['road_relation'][surface_street]}\ncross_walk: {TagDict['road_relation'][cross_walk]}\n"
        inter_actor_all = actor_state.agent_tag['inter_actor']
        if len(inter_actor_all):
            inter_actor_tag = inter_actor_all[list(inter_actor_all)[0]]['relation'][i]
            inter_actor = str(inter_actor_tag).split(".")[0]
            inter_actor_info = TagDict['inter_actor_relation'][inter_actor]
        else:
            inter_actor_info = 'None'
        ax_title = f"{ax_title}inter_actor: {inter_actor_info}\n"
    handels,labels = [],[]
    ax_handels,ax_labels = ax.get_legend_handles_labels()
    handels.extend(ax_handels)
    labels.extend(ax_labels)
    by_label = OrderedDict(zip(labels,handels))
    ax.legend().remove()
    fig.legend(by_label.values(),by_label.keys(),loc='upper center',ncol=3,fontsize=font1['size'])
    if not FILE.startswith('Signalized'):
        ax.set_xlim(0,150)
        ax.set_ylim(-150,-110)
    else:
        ax.set_xlim(-160,0)
        ax.set_ylim(-150,-110)
    ax.set_title(ax_title)
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    plt.close()
    del fig, ax

def plot_actor_polygon(actor_polygon,step,valid_start,valid_end,ax,polygon_label,color_type,gradient=False):
    colors = get_color_map(ax,0,int(TTC_2*sampling_frequency),gradient)
    if step >= valid_start and step <= valid_end:
        actor_polygon_step = actor_polygon[step]
        if isinstance(actor_polygon_step,list):
            for i,actor_polygon_step_ in enumerate(actor_polygon_step):
                x,y = actor_polygon_step_.exterior.xy
                if gradient:
                    ax.fill(x,y,c=colors[i])
                else:
                    color,transparency = actor_color[color_type]['color'],actor_color[color_type]['alpha']
                    ax.fill(x,y,c=color,alpha=transparency,label=polygon_label)
        elif actor_polygon_step.__class__.__name__ =='Polygon':
            x,y = actor_polygon_step.exterior.xy
            if gradient:
                ax.fill(x,y,c=actor_color[color_type]['color'])
            else:
                color,transparency = actor_color[color_type]['color'],actor_color[color_type]['alpha']
                ax.fill(x,y,c=color,alpha=transparency,label=polygon_label)
        elif actor_polygon_step.__class__.__name__ =='MultiPolygon':
            for i,polygon in enumerate(actor_polygon_step):
                x,y = polygon.exterior.xy
                if gradient:
                    ax.fill(x,y,c=colors[i])
                else:
                    color,transparency = actor_color[color_type]['color'],actor_color[color_type]['alpha']
                    ax.fill(x,y,c=color,alpha=transparency,label=polygon_label)
    else:
        ...
    return ax

def parsing_data(DATADIR,FILE,eval_mode=False):
    if eval_mode:
        parsed = get_parsed_carla_data(DATADIR/FILE)
    else:
        parsed = get_parsed_data(DATADIR/FILE)
    return parsed

def main_plot(args,parsed,FILE,RESULT_DIR,RESULT_FILENAME,fileprefix,FILENUM):
    environment_element = EnvironmentElementsWaymo(parsed)
    original_data_roadgragh,original_data_light = environment_element.road_graph_parser(eval_mode=args.eval_mode)
    environment_element.create_polygon_set(eval_mode=args.eval_mode)
    result = json.load(open(RESULT_DIR / RESULT_FILENAME,'r'))
    actors_list = result['general_info']['actors_list']
    actors_states = namedtuple('actors_states', 'actor_type,agent,agent_state,agent_tag,tp,etp,ebb,v_s,v_e')
    actors_states_list = []
    for actor_type, agents in actors_list.items():
        if isinstance(agents,int):
            agents = [agents]
        for agent in agents:
            agent_state,_ = actor_creator(actor_type,agent,parsed,eval_mode=args.eval_mode)
            val_proportion = agent_state.data_preprocessing()
            v_s,v_e = agent_state.get_validity_range()
            tp = agent_state.polygon_set()
            yaw_rate = result['actors_activity'][actor_type][f"{actor_type}_{agent}_activity"]['yaw_rate']
            _ = agent_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=sampling_frequency,yaw_rate=yaw_rate)
            etp = agent_state.expanded_multipolygon
            ebb = agent_state.expanded_bbox_list(expand=bbox_extension)
            agent_tag = {}
            agent_tag['lo_act'] = result['actors_activity'][actor_type][f"{actor_type}_{agent}_activity"]['lo_act']
            agent_tag['la_act'] = result['actors_activity'][actor_type][f"{actor_type}_{agent}_activity"]['la_act']
            agent_tag['environment'] = {}
            agent_tag['environment']['surface_street'] = result['actors_environment_element_intersection'][actor_type][f"{actor_type}_{agent}"]['surface_street']['relation']
            agent_tag['environment']['cross_walk'] = result['actors_environment_element_intersection'][actor_type][f"{actor_type}_{agent}"]['cross_walk']['relation']
            agent_tag['inter_actor'] = result['inter_actor_relation'][f"{actor_type}_{agent}"]
            agent_states = actors_states(actor_type,agent,agent_state,agent_tag,tp,etp,ebb,v_s,v_e)
            actors_states_list.append(agent_states)
    time_steps = actors_states_list[0].agent_state.time_steps
    for i in track(range(time_steps)):
        save_path = ROOT / "results" / "gp1" / "eval" / f"{fileprefix}-{FILENUM}"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"{i}.jpg"
        plot_single_image(parsed,FILE,
                            actors_states_list,
                            i,time_steps,
                            environment_element,
                            original_data_roadgragh,
                            original_data_light,
                            save_path,eval_mode=args.eval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', action="store_true" ,help='[bool] True for evaluation mode')
    parser.add_argument('--result_time', type=str, required=True, help='#result time to plot.e.g.:11-09-09_25')
    parser.add_argument('--filenum', type=str, required=False, help='specify one file to plot.e.g.:00003')
    args = parser.parse_args()
    if args.eval_mode and not args.filenum:
        DATA_DIR = ROOT / "waymo_open_dataset" / "data" / "eval_data" / "carla_data"
        RESULT_DIR = ROOT / "results" / "gp1" / f"2023-{args.result_time}"
        for FILE_PATH in DATA_DIR.iterdir():
            FILE = FILE_PATH.name
            if not FILE.endswith(".pkl"):
                continue
            FILENUM = re.search(r"-(\d{5})-", FILE)
            if FILENUM is not None:
                FILENUM = FILENUM.group()[1:-1]
                print(f"Processing file: {FILE}")
            else:
                print(f"File name error: {FILE}")
                continue
            fileprefix = FILE.split('-')[0]
            RESULT_FILENAME = f'{fileprefix}_{FILENUM}_2023-{args.result_time}_tag.json'
            parsed = parsing_data(DATA_DIR,FILE,eval_mode=args.eval_mode)
            main_plot(args,parsed,FILE,RESULT_DIR,RESULT_FILENAME,fileprefix,FILENUM)


    elif  args.eval_mode and args.filenum:
        DATADIR = ROOT / "waymo_open_dataset" / "data" / "eval_data" / "carla_data"
        RESULT_DIR = ROOT / "results" / "gp1" / f"2023-{args.result_time}"
        fileprefix = "SignalizedJunctionLeftTurn"
        FILE = f"{fileprefix}-{args.filenum}-of-00010.pkl"
        RESULT_FILENAME = f'{fileprefix}_{args.filenum}_2023-{args.result_time}_tag.json'
        parsed = parsing_data(DATADIR,FILE,eval_mode=args.eval_mode)
        main_plot(args,parsed,FILE,RESULT_DIR,RESULT_FILENAME,fileprefix,args.filenum)
    else:
        DATADIR = ROOT / "waymo_open_dataset" / "data" / "tf_example" / "training"