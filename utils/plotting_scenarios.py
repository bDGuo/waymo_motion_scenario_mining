"""
generate figures for scenarios
Author:Detian Guo
Date: 04/11/2022
"""
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import json
import os
import numpy as np
from helpers.os_helpers import *
from tags_generation import generate_lanes,road_graph_parser
from helpers.diverse_plot import plot_road_lines,create_figure_and_axes
from static_elements import StaticElementsWaymo
from create_rect_from_file import rect_object_creator
from tags_dict import lo_act_dict,la_act_dict

font1 = {'family' : 'Times New Roman','weight' : 'normal','size':20}
font2 = {'family' : 'Times New Roman','weight' : 'normal','size':30}
lane_color = {
    'freeway':'k',
    'surface_street':'slategray',
    'bike_lane':'maroon',
    'cross_walk':'lightgray',
    'speed_bump':'darkgoldenrod'
}
actor_color = {
    'host_a':{'color':'r','alpha':0.5},
    'host_e':{'color':'b','alpha':0.3},
    'guest_a':{'color':'yellow','alpha':0.3},
    'guest_e':{'color':'green','alpha':0.1}
}

size_pixels = 1000
lane_key = ['freeway','surface_street','bike_lane']
other_object_key = ['cross_walk','speed_bump']
bbox_extension = 2
# parameter for estimation of the actor approaching a static element
TTC_1 = 5
# parameter for estimation of two actors' interaction
TTC_2 = 9

def plot_all_scenarios(DATADIR,FILE,FILENUM,RESULT_DIR,RESULT_FILENAME,RESULT_SOLO,FIGUREDIR):
    fig_file_path = mkdir(FIGUREDIR,FILENUM)
    result = json.load(open(os.path.join(RESULT_DIR,RESULT_FILENAME),'r'))
    solo_scenarios = json.load(open(os.path.join(RESULT_DIR,RESULT_SOLO),'r'))
    actors_list = result['actors_list']
    inter_actor_relation = result['inter_actor_relation']
    actors_activity = result['actors_activity']
    for actor_type,agents in actors_list.items():
        for agent in agents:
            # ###############################
            # if agent!=0:
            #     continue
            # ###############################
            agent_activity = actors_activity[actor_type][f"{actor_type}_{agent}_activity"]
            agent_interalation = inter_actor_relation[f"{actor_type}_{agent}"]
            agent_fig_path = mkdir(fig_file_path,f"{actor_type}_{agent}")
            agent_state,_ = rect_object_creator(actor_type,agent,DATADIR,FILE)
            validity_proportion = agent_state.data_preprocessing()
            solo_scenario = solo_scenarios[actor_type][f"{actor_type}_{agent}"]
            _=plot_solo_scenario(f"{actor_type}_{agent}",agent_activity,agent_interalation,agent_state,DATADIR,FILE,solo_scenario,agent_fig_path)
    
    return 0

def plot_solo_scenario(agent,agent_activity,agent_interalation,agent_state,DATADIR,FILE,solo_scenario,agent_fig_path):
    """
    plot the scenarios for one agent
    """
    #################################
    ######  the first figure #######
    #################################
    valid_start,valid_end = agent_state.get_validity_range()
    nrows,ncols = 1,2
    plt.rc('font',family='Times New Roman',size=font1['size'])
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*8,nrows*5))
    ax_list = axes.flatten() #type:ignore
    # Plot longitudinal velocity and activity
    ax_list[0] = plot_actor_activity(agent_activity["long_v"],solo_scenario["lo"],\
        valid_start,valid_end,ax_list[0],"Longitudinal velocity [m/s]","Longitudinal activity [-]","Longitudinal")
    # Plot longitudinal velocity and activity
    ax_list[1] = plot_actor_activity(agent_activity["yaw_rate"],solo_scenario["la"],\
        valid_start,valid_end,ax_list[1],"Yaw rate[rad/s]","Lateral activity [-]","Lateral")
    plt.tight_layout()
    plt.savefig(f"{agent_fig_path}\{agent}_activity.jpg",bbox_inches="tight")
    #################################
    ######  the second figure #######
    #################################
    # plot colorful actual trajectory
    nrows,ncols = 1,4
    plt.rc('font',family='Times New Roman',size=font2['size'])
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*15,nrows*15))
    ax_list = axes.flatten() #type:ignore
    actor_trajectory_polygon = agent_state.polygon_set()
    ax_list[0],s_e,o_d_r,o_d_l = plot_road_graph(DATADIR,FILE,ax=ax_list[0])
    ax_list[0] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list[0],"Actual trajectory",gradient=True,host=True,type_a=True)
    handels,labels = ax_list[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list[0].legend(by_label.values(),by_label.keys(),loc="upper right",prop=font2)
    ax_list[0] = set_scaling_2(ax_list[0],agent_state,valid_start,valid_end)
    # plot relation with static elements
    _ = agent_state.expanded_polygon_set(TTC=TTC_1,sampling_fq=10)
    actor_expanded_multipolygon = agent_state.expanded_multipolygon
    ax_list[1],_,_,_ = plot_road_graph(DATADIR,FILE,ax=ax_list[1],static_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l)
    ax_list[1] = plot_actor_polygons(actor_expanded_multipolygon,valid_start,valid_end,ax_list[1],"Extended trajectory host",gradient=False,host=True,type_a=False)
    ax_list[1] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list[1],"Actual trajectory host",gradient=False,host=True,type_a=True)
    handels,labels = ax_list[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list[1].legend(by_label.values(),by_label.keys(),loc="upper right",prop=font2)
    # ax_list[1] = set_scaling(ax_list[1])
    # extended trajectory pologons
    etp = agent_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=10)
    # generate the extended bounding boxes
    ebb = agent_state.expanded_bbox_list(expand=bbox_extension)
    # plot band relations type 1
    ax_list[2],_,_,_ = plot_road_graph(DATADIR,FILE,ax=ax_list[2],static_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l)
    # plot band relations type 2
    ax_list[3],_,_,_ = plot_road_graph(DATADIR,FILE,ax=ax_list[3],static_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l)
    ax_list[2] = plot_actor_polygons(etp,valid_start,valid_end,ax_list[2],f"ETP host:{agent}",gradient=False,host=True,type_a=False)
    ax_list[3] = plot_actor_polygons(ebb,valid_start,valid_end,ax_list[3],f"EBB host:{agent}",gradient=False,host=True,type_a=False)
    ax_list[2] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list[2],f"Actual host:{agent}",gradient=False,host=True,type_a=True)
    ax_list[3] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list[3],f"Actual host:{agent}",gradient=False,host=True,type_a=True)
    actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}
    for key in agent_interalation:
        guest_type,guest_id = key.split("_")
        guest_state,_ = rect_object_creator(actor_dict[guest_type],int(guest_id),DATADIR,FILE)
        validity_proportion = guest_state.data_preprocessing()
        guest_trajectory_polygon = guest_state.polygon_set()
        # extended trajectory pologons
        guest_etp = guest_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=10)
        # generate the extended bounding boxes
        guest_ebb = guest_state.expanded_bbox_list(expand=bbox_extension)
        guest_v_s,guest_v_e = guest_state.get_validity_range()
        ax_list[2] = plot_actor_polygons(guest_etp,guest_v_s,guest_v_e,ax_list[2],f"ETP guest:{guest_type}",gradient=False,host=False,type_a=False)
        ax_list[3] = plot_actor_polygons(guest_ebb,guest_v_s,guest_v_e,ax_list[3],f"EBB guset:{guest_type}",gradient=False,host=False,type_a=False)
        ax_list[2] = plot_actor_polygons(guest_trajectory_polygon,guest_v_s,guest_v_e,ax_list[2],f"Actual guest:{guest_type}",gradient=False,host=False,type_a=True)
        ax_list[3] = plot_actor_polygons(guest_trajectory_polygon,guest_v_s,guest_v_e,ax_list[3],f"Actual guest:{guest_type}",gradient=False,host=False,type_a=True)
    handels,labels = ax_list[2].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list[2].legend(by_label.values(),by_label.keys(),prop=font2,ncol=1)
    handels,labels = ax_list[3].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list[3].legend(by_label.values(),by_label.keys(),prop=font2,ncol=1)
    for i in range(len(ax_list)):
        ax_list[i].tick_params(labelsize=font2['size']) 
    plt.tight_layout()
    plt.savefig(f"{agent_fig_path}\{agent}_roadgraph.jpg",bbox_inches="tight")
    return 0

def plot_road_graph(DATADIR,FILE,ax,static_element=None,original_data_roadgragh=None,original_data_light=None):
    if not static_element or not original_data_roadgragh or not original_data_light:
        original_data_roadgragh,original_data_light = road_graph_parser(DATADIR,FILE)
        static_element = StaticElementsWaymo(original_data_roadgragh,original_data_light)
        static_element.create_polygon_set()
    # plot lanes
    for lane_type in lane_key:
        lane_polygon_set = static_element.get_lane(lane_type)
        for lane_polygon in lane_polygon_set:
                x,y = lane_polygon.exterior.xy
                ax.fill(x,y,c=lane_color[lane_type],label=f'{lane_type}')
    # plot other type
    for other_object_type in other_object_key:
        other_object_polygon_list = static_element.get_other_object(other_object_type)
        for other_object_polygon in other_object_polygon_list:
            x,y = other_object_polygon.exterior.xy
            ax.fill(x,y,c=lane_color[other_object_type],label=f'{other_object_type}')
    # plot road lines
    ax = plot_road_lines(ax,original_data_roadgragh,original_data_light,road_lines=1) #type:ignore
    ax.set_xlabel('x(m)',fontdict=font2)
    ax.set_ylabel('y(m)',fontdict=font2)
    ax.tick_params(labelsize=font2['size']*0.8)
    plt.xticks(fontname = "Times New Roman")
    plt.yticks(fontname = "Times New Roman")
    ax.set_aspect('equal')
    # handels,labels = ax.get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels,handels))
    # ax.legend(by_label.values(),by_label.keys(),loc="upper right",markerscale=8.0,prop=font1)
    return ax,static_element,original_data_roadgragh,original_data_light

def plot_actor_polygons(actor_polygon:list,valid_start:int,valid_end:int,ax,polygon_label:str,gradient:bool=False,host:bool=True,type_a:bool=True):
    colors = get_color_map(ax,valid_start,valid_end,gradient)
    for step in range(valid_start,valid_end+1):
        actor_polygon_step = actor_polygon[step]
        if isinstance(actor_polygon_step,list):
            for actor_polygon_step_ in actor_polygon_step:
                x,y = actor_polygon_step_.exterior.xy
                if gradient:
                    ax.fill(x,y,c=colors[step-valid_start])
                else:
                    if host and type_a:
                        color,transparency = actor_color['host_a']['color'],actor_color['host_a']['alpha']
                    elif host and not type_a:
                        color,transparency = actor_color['host_e']['color'],actor_color['host_e']['alpha']
                    elif not host and type_a:
                        color,transparency = actor_color['guest_a']['color'],actor_color['guest_a']['alpha']
                    else:
                        color,transparency = actor_color['guest_e']['color'],actor_color['guest_e']['alpha']
                    ax.fill(x,y,c=color,alpha=transparency,label=polygon_label)
        else:
            x,y = actor_polygon_step.exterior.xy
            if gradient:
                ax.fill(x,y,c=colors[step-valid_start])
            else:
                if host and type_a:
                    color,transparency = actor_color['host_a']['color'],actor_color['host_a']['alpha']
                elif host and not type_a:
                    color,transparency = actor_color['host_e']['color'],actor_color['host_e']['alpha']
                elif not host and type_a:
                    color,transparency = actor_color['guest_a']['color'],actor_color['guest_a']['alpha']
                else:
                    color,transparency = actor_color['guest_e']['color'],actor_color['guest_e']['alpha']
                ax.fill(x,y,c=color,alpha=transparency,label=polygon_label)
    return ax

def plot_actor_activity(data,activity,valid_start,valid_end,ax1,legend_data:str,legend_activity:str,title:str):
    # sampling frequency = 10
    if not isinstance(valid_start,int):
        valid_start = int(valid_start)
        valid_end = int(valid_end)
    time = np.arange(len(data)) / 10
    color_map = get_color_map(ax1,-5,2)
    if legend_data.startswith("Long"):
        act_dict = {v:k for k,v in lo_act_dict.items()}
    else:
        act_dict = {v:k for k,v in la_act_dict.items()}
    ax1.plot(time[valid_start:valid_end+1],data[valid_start:valid_end+1],c='gray',linewidth=15,alpha=0.3,label='valid range')
    for _,values in activity.items():
        if not isinstance(values['start'],int):
            values['start'] = int(values['start'])
            values['end'] = int(values['end'])
        color  = int(act_dict[values['event']])
        ax1.plot(time[values['start']:values['end']+1],\
            data[values['start']:values['end']+1],c=color_map[color],label=values['event'],linewidth=3)
    ax1.set_xticks(np.arange(0,np.max(time),1),fontname = "Times New Roman",fontsize=font1['size'])
    xlabel = 'time (s)'
    ax1.grid()
    handels,labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax1.legend(by_label.values(),by_label.keys(),markerscale=8.0,prop=font1)
    ax1.set_xlabel(xlabel,fontdict=font1)
    ax1.set_ylabel(legend_data,fontdict=font1)
    return ax1

def plot_actor_activity_2(data,activity,valid_start,valid_end,ax1,ax2,legend_data:str,legend_activity:str,title:str):
    # sampling frequency = 10
    if not isinstance(valid_start,int):
        valid_start = int(valid_start)
        valid_end = int(valid_end)
    time = np.arange(len(data)) / 10
    ax1.plot(time,data,'r-',label=legend_data)
    ax2.plot(time,activity,'b--',label=legend_activity)
    ax1.plot(time[valid_start:valid_end+1],data[valid_start:valid_end+1],c='gray',linewidth=15,alpha=0.3,label='valid range')
    ax2.plot(time[valid_start:valid_end+1],activity[valid_start:valid_end+1],c='gray',linewidth=15,alpha=0.3,label='valid range')
    ax2.set_yticks(np.arange(-6,3,1),fontname = "Times New Roman",fontsize=font1['size'])
    ax1.set_xticks(np.arange(0,np.max(time),1),fontname = "Times New Roman",fontsize=font1['size'])
    ax2.set_xticks(np.arange(0,np.max(time),1),fontname = "Times New Roman",fontsize=font1['size'])
    xlabel = 'time (s)'
    ax1.grid()
    ax2.grid()
    ax1.legend(prop=font1)
    ax2.legend(prop=font1)
    if legend_data == "Long. velocity":
        legend_data = f"{legend_data} [m/s]"
    else:
        legend_data = f"{legend_data} [rad/s]"
    ax1.set_xlabel(xlabel,fontdict=font1)
    ax2.set_xlabel(xlabel,fontdict=font1)
    ax1.set_ylabel(legend_data,fontdict=font1)
    ax2.set_ylabel(legend_activity,fontdict=font1)
    return ax1,ax2

def get_color_map(ax,valid_start,valid_end,gradient:bool=False,plot:bool=False):

    if gradient:
        vs = np.linspace(valid_start,valid_end,valid_end-valid_start+1)
        vs = vs / 10
        norm = plt.Normalize(valid_start/10,valid_end/10) #type:ignore
        color_map = plt.cm.jet #type:ignore
        # plot one agent trajectory with rectangualrs
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm) #type:ignore
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label('time [s]',fontfamily=font2['family'],fontsize=font2['size'])
        cb.set_ticks(np.linspace(np.min(vs),np.max(vs),9))
        ticks = cb.get_ticks()
        cblabels = np.linspace(valid_start,valid_end,len(ticks))/10
        cblabels = [f"{i:.1f}" for i in cblabels]
        cb.set_ticks(ticks,labels=cblabels,fontfamily=font2['family'],fontsize=font2['size'])
    else:
        vs = np.linspace(valid_start,valid_end,valid_end-valid_start+1)
        norm = plt.Normalize(valid_start,valid_end) #type:ignore
        color_map = plt.cm.jet #type:ignore
        # plot one agent trajectory with rectangualrs
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm) #type:ignore
        sm.set_array([])
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm) #type:ignore
    colors = color_map(norm(vs))
    return colors

def set_scaling(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim,ylim = get_scaling(xlim,ylim,5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def get_scaling(xlim,ylim,factor):
    center_y = (ylim[-1] + ylim[0]) / 2
    center_x = (xlim[-1] + xlim[0]) / 2
    range_y = ylim[-1] - ylim[0]
    range_x = xlim[-1] - xlim[0]
    width = max(range_y,range_x)
    xlim = [center_x - width*factor/2,center_x + width*factor/2]
    ylim = [center_y - width*factor/2,center_y + width*factor/2]
    return xlim,ylim

def set_scaling_2(ax,agent_state,valid_start,valid_end):
    position_x = agent_state.kinematics["x"].numpy().squeeze()[valid_start:valid_end+1]
    position_y = agent_state.kinematics["y"].numpy().squeeze()[valid_start:valid_end+1]
    xlim,ylim = get_scaling([np.min(position_x),np.max(position_x)],[np.min(position_y),np.max(position_y)],40)
    xlim_ = ax.get_xlim()
    ylim_ = ax.get_ylim()
    xlim = [max(xlim[0],xlim_[0]),min(xlim[1],xlim_[1])]
    ylim = [max(ylim[0],ylim_[0]),min(ylim[1],ylim_[1])]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax








