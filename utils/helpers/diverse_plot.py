
import uuid
from cmath import pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Rectangle


def create_figure_and_axes(size_pixels:int,axes_x=1,axes_y=1):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(axes_x,axes_y, num=uuid.uuid4())
    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches(size_inches, size_inches)
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    fig.set_tight_layout(True)
    return fig, ax

def get_viewport(all_states,all_states_mask):
    """
    Gets the region containing the data.

    Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]
    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2
    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)
    width = max(range_y, range_x)
    return center_y, center_x, width

def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def get_colormap(num_steps:int):
    """Compute a color map array of shape [num_steps, 4]."""
    colors = cm.get_cmap('jet', num_steps)
    colors = colors(np.arange(num_steps))
    np.random.shuffle(colors)
    return colors

def visualize_one_agent(states,
                        mask,
                        center_y,
                        center_x,
                        color_map,
                        fig,
                        ax,
                        agent_type,
                        width,
                        linewidth=1,
                        traj=True,
                        ):
    """Generate visualization for all steps. num_agents = states[0,:]"""

    masked_x = states[:,0][mask]
    masked_y = states[:,1][mask]
    masked_angle = states[:,2][mask]
    masked_l = states[:,3][mask]
    masked_w = states[:,4][mask]
    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    if traj:
    # Plot one agent trajectory with marker 'o'
        ax.scatter(
            masked_x,
            masked_y,
            marker='o',
            linewidths=linewidth
        )
    else:
    # plot one agent trajectory with rectangualrs 
        for i,(x,y,angle,c) in enumerate(zip(masked_x,masked_y,masked_angle,color_map)):
            rect = Rectangle((x,y),masked_l[i],masked_w[i],angle=angle/pi*180,color=c)
            ax.add_patch(rect)
    # ax.text(masked_x[0],masked_y[0],f"{agent_type}")
    image = fig_canvas_image(fig)
    return 0

def plot_road_lines(ax,original_data_roadgragh:dict,original_data_light:dict,road_edge:bool=True,road_lines:bool=False,lane_center:bool=False,controlled_lane:bool=False):
    
    roadgraph_type = original_data_roadgragh['roadgraph_type']
    roadgraph_xyz = original_data_roadgragh['roadgraph_xyz']
    roadgraph_lane_id = original_data_roadgragh['roadgraph_lane_id']
    traffic_lights_id = original_data_light['traffic_lights_id']
    traffic_lights_valid = original_data_light['traffic_lights_valid']
    lane_type = {
    'freeway':1,
    'surface_street':2,
    'bike_lane':3
    }
    # road edges
    if road_edge:
        roadedge_mask = np.where((roadgraph_type[:,0]==15) | (roadgraph_type[:,0]==16))[0]
        roadedge_pts = roadgraph_xyz[roadedge_mask,:2].T
        if (len(roadedge_pts[0,:])>0):
            ax.scatter(roadedge_pts[0,:],roadedge_pts[1,:],color='k',marker='.',s=20,label="road edge points")

    # road lines
    if road_lines:
        roadline_mask = np.where((roadgraph_type[:,0]>=6) & (roadgraph_type[:,0]<=13))[0]
        roadline_pts = roadgraph_xyz[roadline_mask,:2].T
        if (len(roadline_pts[0,:])>0):
            ax.scatter(roadline_pts[0,:],roadline_pts[1,:],color='k',marker='.',s=20,label="road line points")

    if lane_center:
    # plot lane center
        for key in lane_type:
            lane_mask = np.where(roadgraph_type[:,0]==lane_type[key])[0]
            lane_pts = roadgraph_xyz[lane_mask,:2].T
            if len(lane_pts[0,:]):
                ax.scatter(lane_pts[0,:],lane_pts[1,:],color='k',marker=".",s=20,label=f"lane center: {key}")

    # plot controlled lanes
    controlled_lanes_id = np.unique(traffic_lights_id[traffic_lights_valid==1])
    if len(controlled_lanes_id) and controlled_lane:
        for controlled_lane_id in controlled_lanes_id:
            controlled_lanes_pts = np.where(roadgraph_lane_id==controlled_lane_id)[0]
            if(len(controlled_lanes_pts)):
                ax.scatter(roadgraph_xyz[controlled_lanes_pts,0],roadgraph_xyz[controlled_lanes_pts,1],color='g',marker=".",s=20,label=f"controlled lane points")
    return ax