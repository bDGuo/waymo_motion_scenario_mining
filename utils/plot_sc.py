from pathlib import Path
import numpy as np
import json
import argparse
from helpers.create_rect_from_file import get_agent_list, actor_creator, get_parsed_data, get_parsed_carla_data
from environ_elements import EnvironmentElementsWaymo
import matplotlib.pyplot as plt
from parameters.plot_parameters import font1, font2
from parameters.plot_parameters import *
from parameters.tag_parameters import *
from plotting_scenarios import plot_road_graph,set_scaling_3
from helpers.diverse_plot import plot_road_lines
from collections import OrderedDict
from logger.logger import *
from rich.progress import track

ROOT = Path(__file__).parent.parent
DATA_DIR = Path("E:/VRU_prediction_dataset/waymo")
SC_ID = ["SC7","SC13","SC1"] # corresponding to SC1, SC3, SC2 in the paper
actor_type = {
    "SC1":[1,1],
    "SC7":[1,3],
    "SC13":[1,2]
}

def parse_json(RESULT_DIR):
    for sc_id in track(SC_ID,description="Plotting..."):
        SC_DIR = RESULT_DIR / sc_id
        host_type, guest_type = actor_type[sc_id][0], actor_type[sc_id][1]
        for sc_file in SC_DIR.glob("*.json"):
            data_num = sc_file.name.split("_")[1]
            DATA = f"training_tfexample.tfrecord-{data_num}-of-01000"
            try:
                parsed = get_parsed_data(DATA_DIR / DATA)
                with open(sc_file,"r") as f:
                    sc_data = json.load(f)
                    for k,v in sc_data.items():
                        host_index = v["host_actor"]
                        guest_index = int(v["guest_actor"])
                        time = np.array(v["time_stamp"])
                        if len(time)==1:
                            continue
                        host,guest = (host_type,host_index),(guest_type,guest_index)
                        plot_single_scenario(parsed,host,guest,time, sc_id, data_num, k)
            except Exception as e:
                logger.error(f"Error in {DATA_DIR / DATA}.")
                logger.error(e)

def plot_actor_traj(ax,actor,color,time,label:str):
    traj_x,traj_y = actor.kinematics["x"],actor.kinematics["y"]
    yaw = actor.kinematics["bbox_yaw"][time[0]]
    length,width = actor.kinematics["length"][time[0]],actor.kinematics["width"][time[0]]
    init_polygon = actor.instant_polygon(traj_x[time[0]],traj_y[time[0]],yaw,length,width)
    init_plg_x,init_plg_y = init_polygon.exterior.xy
    ax.fill(init_plg_x,init_plg_y,color=color,alpha=0.5,label=f"{label}'s initial bounding box")
    dx = traj_x[time[-1]] - traj_x[time[-2]]
    dy = traj_y[time[-1]] - traj_y[time[-2]]
    arrow=plt.arrow(traj_x[time[-2]], traj_y[time[-2]], dx, dy, width=0.3,head_width=2,color=color, label=f"{label}'s trajectory")
    ax.plot(traj_x[time], traj_y[time], color=color, linewidth=4, linestyle="-")
    return ax

def plot_scenario(parsed, host_actor, guest_actor, time):
    nrows, ncols = 1, 1
    plt.rc('font', **font2)
    fig,ax = plt.subplots(nrows, ncols, figsize=(10, 10))
    ax.set_aspect('equal')
    environment_element = EnvironmentElementsWaymo(parsed)
    original_data_roadgragh,original_data_light = environment_element.road_graph_parser()
    # ax,environment_element,original_data_roadgragh,original_data_light = plot_road_graph(parsed, ax)
    ax = plot_road_lines(ax, original_data_roadgragh,original_data_light,road_lines=True,controlled_lane=True)
    ax = plot_actor_traj(ax, host_actor, 'r', time,"host")
    ax = plot_actor_traj(ax, guest_actor, 'b', time,"guest")
    ax = set_scaling_3(ax, host_actor, guest_actor,time[0], time[-1])
    handels,labels = [],[]
    ax_handels,ax_labels = ax.get_legend_handles_labels()
    handels.extend(ax_handels)
    labels.extend(ax_labels)
    by_label = OrderedDict(zip(labels,handels))
    axbox = ax.get_position()
    ax.legend(by_label.values(),by_label.keys(),ncol=2,
              bbox_to_anchor=(0., 1.0, 1., 0.), loc='lower left',
              markerscale=15,
              prop=font1)
    return fig,ax

def plot_single_scenario(parsed, host:tuple, guest:tuple, time, sc_id, data_num, sc_num):
    FIG_PATH = ROOT / "figures" / "ITSC" / sc_id
    if not FIG_PATH.exists():
        FIG_PATH.mkdir(exist_ok=True, parents=True)
    host_actor,_ = actor_creator(host[0], host[1], parsed)
    guest_actor,_ = actor_creator(guest[0], guest[1], parsed)
    fig,ax = plot_scenario(parsed, host_actor, guest_actor, time)
    plt.savefig(FIG_PATH/f"data{data_num}_num{sc_num}.png",dpi=300)
    plt.clf()
    plt.close()
    return 0

if __name__ == "__main__":
    RESULT_TIME = '2023-04-13-16_22'
    RESULT_DIR = ROOT / "results" / "gp1" / RESULT_TIME
    try:
        parse_json(RESULT_DIR)
    except Exception as e:
        logger.error(e)
        logger.error("Error occurred when plotting scenario")

