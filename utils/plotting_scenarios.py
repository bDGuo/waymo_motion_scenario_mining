"""
generate figures for scenarios
Author:Detian Guo
Date: 04/11/2022
"""
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from environ_elements import EnvironmentElementsWaymo
from helpers.create_rect_from_file import actor_creator, get_parsed_data, get_parsed_carla_data
from helpers.diverse_plot import plot_road_lines
from parameters.plot_parameters import *
from parameters.tag_parameters import *
from parameters.tags_dict import lo_act_dict, la_act_dict


def plot_all_scenarios(DATADIR,FILE,FILENUM,RESULT_DIR,RESULT_FILENAME,RESULT_SOLO,FIGUREDIR,eval_mode=False):
    FIG_PATH = FIGUREDIR / FILENUM
    if not FIG_PATH.exists():
        FIG_PATH.mkdir(exist_ok=True, parents=True)
    result = json.load(open(RESULT_DIR / RESULT_FILENAME,'r'))
    solo_scenarios = json.load(open(RESULT_DIR/RESULT_SOLO,'r'))
    actors_list = result['general_info']['actors_list']
    inter_actor_relation = result['inter_actor_relation']
    actors_activity = result['actors_activity']
    if eval_mode:
        parsed = get_parsed_carla_data(DATADIR/FILE)
    else:
        parsed = get_parsed_data(DATADIR/FILE)
    environment_element = EnvironmentElementsWaymo(parsed)
    original_data_roadgragh,original_data_light = environment_element.road_graph_parser(eval_mode=eval_mode)
    environment_element.create_polygon_set(eval_mode=eval_mode)
    for actor_type,agents in actors_list.items():
        if isinstance(agents,int):
            agents = [agents]
        for agent in agents:
            agent_activity = actors_activity[actor_type][f"{actor_type}_{agent}_activity"]
            agent_interalation = inter_actor_relation[f"{actor_type}_{agent}"]
            AGENT_FIG_PATH = FIG_PATH / f"{actor_type}_{agent}"
            if not AGENT_FIG_PATH.exists():
                AGENT_FIG_PATH.mkdir()
            agent_state,_ = actor_creator(actor_type,agent,parsed,eval_mode=eval_mode)
            val_proportion = agent_state.data_preprocessing()
            solo_scenario = solo_scenarios[actor_type][f"{actor_type}_{agent}"]
            _=plot_solo_scenario(f"{actor_type}_{agent}",agent_activity,actors_activity,agent_interalation,agent_state,parsed,solo_scenario,AGENT_FIG_PATH,\
                environment_element,original_data_roadgragh,original_data_light,eval_mode=eval_mode)
    return 0

def plot_solo_scenario(agent,agent_activity,actors_activity,agent_interalation,agent_state,parsed,solo_scenario,AGENT_FIG_PATH,s_e,o_d_r,o_d_l,eval_mode=False):
    """
    plot the scenarios for one agent
    """
    #################################
    ######  the first figure #######
    #################################
    valid_start,valid_end = agent_state.get_validity_range()
    nrows,ncols = 1,2
    plt.rc('font',family='Times New Roman',size=font1['size'])
    fig,axes1 = plt.subplots(nrows,ncols,figsize=(ncols*8,nrows*5))
    ax_list1 = axes1.flatten() #type:ignore
    # Plot longitudinal velocity and activity
    ax_list1[0] = plot_actor_activity(agent_activity["long_v"],solo_scenario["lo_act"],\
        valid_start,valid_end,ax_list1[0],fig,"Longitudinal velocity [m/s]","Longitudinal activity [-]","Longitudinal")
    # Plot longitudinal velocity and activity
    ax_list1[1] = plot_actor_activity(agent_activity["yaw_rate"],solo_scenario["la_act"],\
        valid_start,valid_end,ax_list1[1],fig,"Yaw rate[rad/s]","Lateral activity [-]","Lateral")
    plt.tight_layout()
    SOLO_ACTIVITY_PATH = AGENT_FIG_PATH / f"{agent}_activity.jpg"
    plt.savefig(SOLO_ACTIVITY_PATH,bbox_inches="tight")
    plt.close()
    #################################
    ######  the second figure #######
    #################################
    actor_trajectory_polygon = agent_state.polygon_set()
    # plot colorful actual trajectory
    nrows,ncols = 1,2
    plt.rc('font',family='Times New Roman',size=font2['size'])
    fig2,axes2 = plt.subplots(nrows,ncols,figsize=(ncols*15,nrows*15))
    ax_list2 = axes2.flatten() #type:ignore
    # extended trajectory pologons
    etp = agent_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=sampling_frequency,yaw_rate=agent_activity["yaw_rate"])
    # generate the extended bounding boxes
    ebb = agent_state.expanded_bbox_list(expand=bbox_extension)
    # plot band relations type 1

    ax_list2[0],_,_,_ = plot_road_graph(parsed,ax=ax_list2[0],environment_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l,eval_mode=eval_mode)
    # plot band relations type 2
    ax_list2[1],_,_,_ = plot_road_graph(parsed,ax=ax_list2[1],environment_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l,eval_mode=eval_mode)
    actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}
    for key in agent_interalation:
        guest_type,guest_id = key.split("_")
        guest_state,_ = actor_creator(actor_dict[guest_type],int(guest_id),parsed,eval_mode=eval_mode)
        _ = guest_state.data_preprocessing()
        guest_trajectory_polygon = guest_state.polygon_set()
        # extended trajectory pologons
        guest_activity = actors_activity[guest_type][f"{guest_type}_{guest_id}_activity"]
        guest_etp = guest_state.expanded_polygon_set(TTC=TTC_2,sampling_fq=sampling_frequency,yaw_rate=guest_activity["yaw_rate"])
        # generate the extended bounding boxes
        guest_ebb = guest_state.expanded_bbox_list(expand=bbox_extension)
        guest_v_s,guest_v_e = guest_state.get_validity_range()
        ax_list2[0] = plot_actor_polygons(guest_etp,guest_v_s,guest_v_e,ax_list2[0],fig2,f"PBB guest",gradient=False,host=False,type_a=False)
        ax_list2[1] = plot_actor_polygons(guest_ebb,guest_v_s,guest_v_e,ax_list2[1],fig2,f"EBB guset",gradient=False,host=False,type_a=False)
        ax_list2[0] = plot_actor_polygons(guest_trajectory_polygon,guest_v_s,guest_v_e,ax_list2[0],fig2,f"Actual guest",gradient=False,host=False,type_a=True)
        ax_list2[1] = plot_actor_polygons(guest_trajectory_polygon,guest_v_s,guest_v_e,ax_list2[1],fig2,f"Actual guest",gradient=False,host=False,type_a=True)
    ax_list2[0] = plot_actor_polygons(etp,valid_start,valid_end,ax_list2[0],fig2,f"PBB host",gradient=False,host=True,type_a=False)
    ax_list2[1] = plot_actor_polygons(ebb,valid_start,valid_end,ax_list2[1],fig2,f"EBB host",gradient=False,host=True,type_a=False)
    ax_list2[0] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list2[0],fig2,f"Actual host",gradient=False,host=True,type_a=True)
    ax_list2[1] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list2[1],fig2,f"Actual host",gradient=False,host=True,type_a=True)
    handels,labels = [],[]
    ax_handels,ax_labels = ax_list2[0].get_legend_handles_labels()
    handels.extend(ax_handels)
    labels.extend(ax_labels)
    ax_handels,ax_labels = ax_list2[1].get_legend_handles_labels()
    handels.extend(ax_handels)
    labels.extend(ax_labels)
    by_label = OrderedDict(zip(labels,handels))
    # ax_list2[0].legend(by_label.values(),by_label.keys(),prop=font2,ncol=1)
    # handels,labels = ax_list2[0].get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels,handels))
    # ax_list2[0].legend(by_label.values(),by_label.keys(),prop=font2,ncol=1)
    axbox = ax_list2[1].get_position()
    ax_list2[0].legend().remove()
    ax_list2[1].legend().remove()
    fig2.legend(by_label.values(),by_label.keys(),ncol=1,
           bbox_to_anchor=(axbox.x0+1.7*axbox.width,axbox.y0+1*axbox.height),markerscale=15)
    plt.tight_layout()
    AGENT_INTERACTOR_PATH = AGENT_FIG_PATH / f"{agent}_inter_actor.jpg"
    plt.savefig(AGENT_INTERACTOR_PATH,bbox_inches="tight")
    plt.close()
    #################################
    ######  the third figure #######
    #################################
    # plot colorful actual trajectory and relation with static elements
    nrows,ncols = 1,2
    plt.rc('font',family='Times New Roman',size=font2['size'])
    fig3,axes3 = plt.subplots(nrows,ncols,figsize=(ncols*15,nrows*15))
    ax_list3 = axes3.flatten() #type:ignore
    ax_list3[0],s_e,o_d_r,o_d_l = plot_road_graph(parsed,ax=ax_list3[0],environment_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l,eval_mode=eval_mode)
    ax_list3[0] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list3[0],fig3,"Actual trajectory",gradient=True,host=True,type_a=True)
    handels,labels = ax_list3[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list3[0].legend(by_label.values(),by_label.keys(),markerscale=15,prop=font2)
    ax_list3[0] = set_scaling_2(ax_list3[0],agent_state,valid_start,valid_end)
    # plot relation with static elements
    actor_expanded_multipolygon = agent_state.expanded_polygon_set(TTC=TTC_1,sampling_fq=sampling_frequency,yaw_rate=agent_activity["yaw_rate"])
    ax_list3[1],_,_,_ = plot_road_graph(parsed,ax=ax_list3[1],environment_element=s_e,original_data_roadgragh=o_d_r,original_data_light=o_d_l,eval_mode=eval_mode)

    ax_list3[1] = plot_actor_polygons(actor_expanded_multipolygon,valid_start,valid_end,ax_list3[1],fig3,"Extended trajectory host",gradient=False,host=True,type_a=False)
    ax_list3[1] = plot_actor_polygons(actor_trajectory_polygon,valid_start,valid_end,ax_list3[1],fig3,"Actual trajectory host",gradient=False,host=True,type_a=True)
    handels,labels = ax_list3[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax_list3[1].legend(by_label.values(),by_label.keys(),markerscale=15,prop=font2)
    plt.tight_layout()
    AGENT_ROADGRAPH_PATH = AGENT_FIG_PATH / f"{agent}_roadgraph.jpg"
    plt.savefig(AGENT_ROADGRAPH_PATH,bbox_inches="tight")
    plt.close()
    return 0

def plot_road_graph(parsed:dict,ax,environment_element=None,original_data_roadgragh=None,original_data_light=None,eval_mode=False,*kwargs):
    '''
    
    kwargs:

    '''
    
    if not environment_element or not original_data_roadgragh or not original_data_light:
        environment_element = EnvironmentElementsWaymo(parsed)
        original_data_roadgragh,original_data_light = environment_element.road_graph_parser(eval_mode=eval_mode)
        environment_element.create_polygon_set(eval_mode=eval_mode)
    # plot lanes
    for lane_type in lane_key:
        lane_polygon_set = environment_element.get_lane(lane_type)
        for lane_polygon in lane_polygon_set:
            x,y = lane_polygon.exterior.xy
            ax.fill(x,y,c=lane_color[lane_type],label=f'{lane_type}')
    # plot other type
    for other_object_type in other_object_key:
        other_object_polygon_list = environment_element.get_other_object(other_object_type)
        for other_object_polygon in other_object_polygon_list:
            x,y = other_object_polygon.exterior.xy
            ax.fill(x,y,c=lane_color[other_object_type],label=f'{other_object_type}')
    # plot road lines
    ax = plot_road_lines(ax,original_data_roadgragh,original_data_light,road_lines=True,controlled_lane=True) #type:ignore
    ax.set_xlabel('X (m)',fontdict=font2)
    ax.set_ylabel('Y (m)',fontdict=font2)
    ax.tick_params(labelsize=font2['size'])
    plt.xticks(fontname = "Times New Roman")
    plt.yticks(fontname = "Times New Roman")
    ax.set_aspect('equal')
    # handels,labels = ax.get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels,handels))
    # ax.legend(by_label.values(),by_label.keys(),loc="upper right",markerscale=15.0,prop=font1)
    return ax,environment_element,original_data_roadgragh,original_data_light

def plot_actor_polygons(actor_polygon,valid_start:int,valid_end:int,ax,fig,polygon_label:str,gradient:bool=False,host:bool=True,type_a:bool=True,inter_actor:bool=False):
    colors = get_color_map(ax,fig,valid_start,valid_end,gradient)
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
        elif actor_polygon_step.__class__.__name__ =='Polygon':
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
        elif actor_polygon_step.__class__.__name__ =='MultiPolygon':
            for polygon in actor_polygon_step.geoms:
                x,y = polygon.exterior.xy
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

def plot_actor_activity(data,activity,valid_start,valid_end,ax1,fig,legend_data:str,legend_activity:str,title:str):
    if not isinstance(valid_start,int):
        valid_start = int(valid_start)
        valid_end = int(valid_end)
    time = np.arange(len(data)) / 10
    color_map = get_color_map(ax1,fig,-5,2)
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
    ax1.set_xticks(np.arange(0,np.max(time)+1,1))
    xlabel = 'time (s)'
    ax1.grid()
    handels,labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handels))
    ax1.legend(by_label.values(),by_label.keys(),markerscale=15.0)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(legend_data)
    return ax1

def plot_actor_activity_2(data,activity,valid_start,valid_end,ax1,ax2,legend_data:str,legend_activity:str,title:str):
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
    if legend_data == "Long. velocity":
        legend_data = f"{legend_data} [m/s]"
    else:
        legend_data = f"{legend_data} [rad/s]"
    return ax1,ax2

def get_color_map(ax,fig,valid_start,valid_end,gradient:bool=False,colorbar:bool=False,cborientation:str="vertical"):
    if gradient:
        vs = np.linspace(valid_start,valid_end,valid_end-valid_start+1)
        # vs = vs / sampling_frequency
        # norm = plt.Normalize(valid_start/sampling_frequency,valid_end/sampling_frequency) #type:ignore
        color_map = cm.get_cmap('Oranges',len(vs)-1)
        norm = mcolors.BoundaryNorm(vs,len(vs)-1)
        # plot one agent trajectory with rectangualrs
        sm = cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        if colorbar:
            cb = fig.colorbar(sm, ax=ax,orientation=cborientation)
            cb.set_label('Time step (-)',fontfamily=font2['family'],fontsize=font2['size'])
            # cb.set_ticks(np.linspace(np.min(vs),np.max(vs),9))
            # ticks = cb.get_ticks()
            # cblabels = np.linspace(valid_start,valid_end,len(ticks))/sampling_frequency
            # cblabels = [f"{i:.2f}" for i in cblabels]
            cb.set_ticks(np.arange(valid_start,valid_end+1,2),labels=np.arange(valid_start,valid_end+1,2),fontfamily=font2['family'],fontsize=font2['size'])
    else:
        vs = np.linspace(valid_start,valid_end,valid_end-valid_start+1)
        color_map = cm.get_cmap('warm',len(vs)-1)
        norm = mcolors.BoundaryNorm(vs,len(vs)-1)
        # plot one agent trajectory with rectangualrs
        sm = cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
    colors = color_map(norm(vs))
    return colors

def set_scaling(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim,ylim = get_scaling(ax,xlim,ylim,5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def get_scaling(ax,xlim,ylim,factor):
    xlim_ = ax.get_xlim()
    ylim_ = ax.get_ylim()
    range_x_ = xlim_[-1] - xlim_[0]
    range_y_ = ylim_[-1] - ylim_[0]
    y_x_ratio = range_y_ / range_x_
    range_y = ylim[-1] - ylim[0]
    range_x = xlim[-1] - xlim[0]
    if not range_y:
        range_y=1.0
    if not range_x:
        range_x=1.0
    if range_x>range_y:
        range_y = range_x * y_x_ratio
    else:
        range_x = range_y / y_x_ratio
    center_y = (ylim[-1] + ylim[0]) / 2
    center_x = (xlim[-1] + xlim[0]) / 2
    xlim = [center_x - range_x*factor/2,center_x + range_x*factor/2]
    ylim = [center_y - range_y*factor/2,center_y + range_y*factor/2]
    range_y = ylim[-1] - ylim[0]
    range_x = xlim[-1] - xlim[0]
    if range_x>range_x_ or range_y>range_y_:
        return xlim_,ylim_
    return xlim,ylim

def set_scaling_2(ax,agent_state,valid_start,valid_end):
    position_x = agent_state.kinematics["x"].squeeze()[valid_start:valid_end+1]
    position_y = agent_state.kinematics["y"].squeeze()[valid_start:valid_end+1]
    xlim,ylim = get_scaling(ax,[np.min(position_x),np.max(position_x)],[np.min(position_y),np.max(position_y)],20)
    # xlim = [max(xlim[0],xlim_[0]),min(xlim[1],xlim_[1])]
    # ylim = [max(ylim[0],ylim_[0]),min(ylim[1],ylim_[1])]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def set_scaling_3(ax,agent_state_1,agent_state_2,valid_start,valid_end):
    position_x_1 = agent_state_1.kinematics["x"].squeeze()[valid_start:valid_end+1]
    position_y_1 = agent_state_1.kinematics["y"].squeeze()[valid_start:valid_end+1]
    position_x_2 = agent_state_2.kinematics["x"].squeeze()[valid_start:valid_end+1]
    position_y_2 = agent_state_2.kinematics["y"].squeeze()[valid_start:valid_end+1]
    position_x = np.concatenate((position_x_1,position_x_2))
    position_y = np.concatenate((position_y_1,position_y_2))
    xlim,ylim = get_scaling(ax,[np.min(position_x),np.max(position_x)],[np.min(position_y),np.max(position_y)],2)
    # xlim = [max(xlim[0],xlim_[0]),min(xlim[1],xlim_[1])]
    # ylim = [max(ylim[0],ylim_[0]),min(ylim[1],ylim_[1])]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax








