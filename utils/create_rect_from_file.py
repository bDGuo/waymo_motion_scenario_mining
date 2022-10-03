"""
Create a rectangular object for testing from a file
"""
import os 
import tensorflow as tf
import argparse
from rect_object import rect_object
from data_parser import features_description
import random
import time


def rect_object_creator(agent_type:int, choice:int):
    """
    description:

    -------------------------------------
    input:
    agent_type: 1 ->vehicle, 2->pedestrian, 3->cyclist
    choice: -1 ->random choice, others -> index 
    -------------------------------------
    output:
    an rect_object instance
    """
    ROOT = os.path.abspath(os.path.dirname("__FILE__"))
    ROOT = os.path.dirname(ROOT)
    DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
    RESULTDIR = os.path.join(ROOT,"results")
    FILE = "training_tfexample.tfrecord-00000-of-01000"
    FILENAME = os.path.join(DATADIR,FILE)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, features_description)

    # a tensor for corresponding agent type
    agent_id_ts = tf.where(parsed['state/type']==agent_type)
    if choice==-1:
        seed = int(time.time())
        random.seed(seed)
        choice = random.randint(0,agent_id_ts.shape[0]-1)
        agent_id = tf.gather(agent_id_ts,choice)
    else:
        agent_id = tf.gather(agent_id_ts,choice)
    
    agent = rect_object(
        {
        'id':tf.gather(parsed['state/id'],agent_id), 
        'type':tf.gather(parsed['state/type'],agent_id),
        'x':tf.gather(tf.concat([parsed['state/past/x'],parsed['state/current/x'],parsed['state/future/x']],1),agent_id),
        'y':tf.gather(tf.concat([parsed['state/past/y'],parsed['state/current/y'],parsed['state/future/y']],1),agent_id),
        'bbox_yaw':tf.gather(tf.concat([parsed['state/past/bbox_yaw'],parsed['state/current/bbox_yaw'],parsed['state/future/bbox_yaw']],1),agent_id),
        'length':tf.gather(tf.concat([parsed['state/past/length'],parsed['state/current/length'],parsed['state/future/length']],1),agent_id),
        'width':tf.gather(tf.concat([parsed['state/past/width'],parsed['state/current/width'],parsed['state/future/width']],1),agent_id),
        'vel_yaw':tf.gather(tf.concat([parsed['state/past/vel_yaw'],parsed['state/current/vel_yaw'],parsed['state/future/vel_yaw']],1),agent_id),
        'velocity_x':tf.gather(tf.concat([parsed['state/past/velocity_x'],parsed['state/current/velocity_x'],parsed['state/future/velocity_x']],1),agent_id),
        'velocity_y':tf.gather(tf.concat([parsed['state/past/velocity_y'],parsed['state/current/velocity_y'],parsed['state/future/velocity_y']],1),agent_id),
        'validity':tf.gather(tf.concat([parsed['state/past/valid'],parsed['state/current/valid'],parsed['state/future/valid']],1),agent_id)
        }
    )
    return agent,choice







