"""
generate figures for evaluation using CARLA data
Author:Detian Guo
Date: 04/11/2022
"""
from collections import namedtuple,OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
from rich.progress import track
import numpy as np
import json
import argparse
from plotting_scenarios import get_color_map,plot_road_graph
from environ_elements import EnvironmentElementsWaymo
from parameters.plot_parameters import *
from parameters.tag_parameters import TTC_2, sampling_frequency,bbox_extension ,time_steps
from helpers.create_rect_from_file import get_agent_list, get_parsed_data, get_parsed_carla_data, actor_creator

# TODO: to be finished for making images used for a video representation of etp
# working directory
ROOT = Path(__file__).parent.parent

def plot_single_image(parsed,actors_states_list:list,i:int, s_e, o_d_r, o_d_l,save_path,eval_mode=False):
    plt.rc('font',family='Times New Roman',size=font2['size'])
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    axes,_,_,_=plot_road_graph(parsed,ax,s_e, o_d_r, o_d_l,eval_mode=eval_mode)
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
    handels,labels = [],[]
    ax_handels,ax_labels = ax.get_legend_handles_labels()
    handels.extend(ax_handels)
    labels.extend(ax_labels)
    by_label = OrderedDict(zip(labels,handels))
    ax.legend().remove()
    fig.legend(by_label.values(),by_label.keys(),loc='upper center',ncol=3,fontsize=font2['size'])
    ax.set_xlim(0,150)
    ax.set_ylim(100,180)
    ax.set_title(f"Time step: {i}")
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    plt.close()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=bool, required=True, default=False ,help='[bool] True for evaluation mode')
    parser.add_argument('--result_time', type=str, required=True, help='#result time to plot.e.g.:11-09-09_25')
    parser.add_argument('--filenum', type=str, required=True, help='#file to plot.e.g.:00003')
    args = parser.parse_args()
    if  args.eval_mode:
        DATADIR = ROOT / "waymo_open_dataset" / "data" / "eval_data" / "carla_data"
        RESULT_DIR = ROOT / "results" / "gp1" / f"2023-{args.result_time}"
        FILE = f"PedestrianCrossing-{args.filenum}-of-00000.pkl"
        RESULT_FILENAME = f'Carla_{args.filenum}_2023-{args.result_time}_tag.json'
        parsed = parsing_data(DATADIR,FILE,eval_mode=args.eval_mode)
        environment_element = EnvironmentElementsWaymo(parsed)
        original_data_roadgragh,original_data_light = environment_element.road_graph_parser(eval_mode=args.eval_mode)
        environment_element.create_polygon_set(eval_mode=args.eval_mode)
        result = json.load(open(RESULT_DIR / RESULT_FILENAME,'r'))
        actors_list = result['general_info']['actors_list']
        actors_states = namedtuple('actors_states', 'actor_type,agent,agent_state,tp,etp,ebb,v_s,v_e')
        actors_states_list = []
        for actor_type, agents in actors_list.items():
            if isinstance(agents,int):
                agents = [agents]
            for agent in agents:
                agent_state,_ = actor_creator(actor_type,agent,parsed,eval_mode=args.eval_mode)
                val_proportion = agent_state.data_preprocessing()
                v_s,v_e = agent_state.get_validity_range()
                tp = agent_state.polygon_set()
                _ = agent_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=sampling_frequency)
                etp = agent_state.expanded_multipolygon
                ebb = agent_state.expanded_bbox_list(expand=bbox_extension)
                agent_states = actors_states(actor_type,agent,agent_state,tp,etp,ebb,v_s,v_e)
                actors_states_list.append(agent_states)
        for i in track(range(time_steps)):
            save_path = ROOT / "results" / "gp1" / "eval" / args.filenum
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f"{i}.jpg"
            plot_single_image(parsed,actors_states_list,i,
                              environment_element,
                              original_data_roadgragh,
                              original_data_light,
                              save_path,eval_mode=args.eval_mode) #type:ignore
    else:
        DATADIR = ROOT / "waymo_open_dataset" / "data" / "tf_example" / "training"