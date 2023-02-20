"""
Create a rectangular object for testing from a file
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from actor import Actor
from .data_parser import features_description
import random
import time

def get_parsed_data(DATADIR,FILE):
    FILENAME = os.path.join(DATADIR,FILE)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, features_description)
    return parsed
    
def get_agent_list(agent_type:int,DATADIR,FILE):
    parsed = get_parsed_data(DATADIR,FILE)
    # a tensor for corresponding agent type
    return tf.where(parsed['state/type']==agent_type).numpy().squeeze()

def actor_creator(agent_type:int, choice:int,DATADIR,FILE):
    """
    description:
    Create dynamical rectangular object from data.
    -------------------------------------
    input:
    agent_type: 1 ->vehicle, 2->pedestrian, 3->cyclist
    choice: -1 ->random choice, others -> index 
    -------------------------------------
    output:
    an rect_object instance
    """
    parsed = get_parsed_data(DATADIR,FILE)

    # a tensor for corresponding agent type
    
    if choice==-1:
        seed = int(time.time())
        random.seed(seed)
        agent_indices_ts = tf.where(parsed['state/type']==agent_type)
        choice = random.randint(0,agent_indices_ts.shape[0]-1)
        agent_index = tf.gather(agent_indices_ts,choice)
    else:
        agent_index = choice
    
    actor_state_dict_tf = __actor_state(parsed,agent_index)

    actor = Actor(actor_state_dict_tf)
    if choice == -1:
        return actor,choice
    return actor,choice

def __actor_state(parsed,agent_index):
    actor_state_dict_tf ={
        'id':tf.gather(parsed['state/id'],agent_index), 
        'type':tf.gather(parsed['state/type'],agent_index),
        'x':tf.gather(tf.concat([parsed['state/past/x'],parsed['state/current/x'],parsed['state/future/x']],1),agent_index),
        'y':tf.gather(tf.concat([parsed['state/past/y'],parsed['state/current/y'],parsed['state/future/y']],1),agent_index),
        'bbox_yaw':tf.gather(tf.concat([parsed['state/past/bbox_yaw'],parsed['state/current/bbox_yaw'],parsed['state/future/bbox_yaw']],1),agent_index),
        'length':tf.gather(tf.concat([parsed['state/past/length'],parsed['state/current/length'],parsed['state/future/length']],1),agent_index),
        'width':tf.gather(tf.concat([parsed['state/past/width'],parsed['state/current/width'],parsed['state/future/width']],1),agent_index),
        'vel_yaw':tf.gather(tf.concat([parsed['state/past/vel_yaw'],parsed['state/current/vel_yaw'],parsed['state/future/vel_yaw']],1),agent_index),
        'velocity_x':tf.gather(tf.concat([parsed['state/past/velocity_x'],parsed['state/current/velocity_x'],parsed['state/future/velocity_x']],1),agent_index),
        'velocity_y':tf.gather(tf.concat([parsed['state/past/velocity_y'],parsed['state/current/velocity_y'],parsed['state/future/velocity_y']],1),agent_index),
        'validity':tf.gather(tf.concat([parsed['state/past/valid'],parsed['state/current/valid'],parsed['state/future/valid']],1),agent_index)
        }
    return actor_state_dict_tf







