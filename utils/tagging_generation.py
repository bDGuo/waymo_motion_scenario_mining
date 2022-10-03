import math
from re import A
import uuid
import time
import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf
import os
from data_parser import features_description

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import itertools

from create_rect_from_file import *
from activity_dectection import *

ROOT = os.path.abspath(os.path.dirname(""))

def agent_id_extraction():

    DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
    RESULTDIR = os.path.join(ROOT,"results")

    FILE = "training_tfexample.tfrecord-00000-of-01000"
    FILENAME = os.path.join(DATADIR,FILE)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, features_description)

    agent_id_ts = {
    "vehicle": tf.where(parsed['state/type']==1),
    "pedestrian": tf.where(parsed['state/type']==2),
    "cyclist": tf.where(parsed['state/type']==3)
    }
    return agent_id_ts

FIGDIR = os.path.join(ROOT,'figures\lo_act')

agent_list = np.array(["","vehicle","pedestrian","cyclist","others"])


def plot_lo_v_act(agent_type,agent_index,SAVEDIR):
    agent_state,agent_id = rect_object_creator(agent_type,agent_index)
    k_h=3
    lo_act = long_act_detector(agent_state,k_h)
    (long_v,_) = agent_state.cordinate_rotate(agent_state.velocity_x,agent_state.velocity_y,agent_state.bbox_yaw)
    long_v = long_v.numpy().squeeze()
    lo_act = lo_act.numpy().squeeze()

    time = np.arange(len(lo_act))
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time,long_v,'r-',label='long_v')
    ax2.plot(time,lo_act,'g--',label='long_act')
    ax1.set_xlabel('time steps')
    ax1.set_ylabel('longitudinal velocity')
    ax2.set_ylabel('longitudinal activity')
    ax2.set_ylim([-6,6])

    plt.grid()
    fig.legend()
    plt.savefig(f"{SAVEDIR}\{agent_list[agent_type]}:{agent_id}.jpg")

agent_id_ts = agent_id_extraction()
for key in agent_id_ts:
    agent_id_np = agent_id_ts[key].numpy().squeeze()
    for agent_id in agent_id_np:
        agent_type = np.where(agent_list==key)[0][0]
        print(agent_type)
        print(agent_id)
        plot_lo_v_act(agent_type,agent_id,FIGDIR)