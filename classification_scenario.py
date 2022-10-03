import math
import uuid
import time
import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf
import os
from utils.data_parser import features_description

START = time.time()

ROOT = os.path.abspath(os.path.dirname(""))
DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
RESULTDIR = os.path.join(ROOT,"results")

# Create Dataset
FILE = "training_tfexample.tfrecord-00000-of-01000"
FILENAME = os.path.join(DATADIR,FILE)
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
data = next(dataset.as_numpy_iterator())
parsed = tf.io.parse_single_example(data, features_description)

from utils.rect_object import *
import json
from rich.progress import track

count_collision_ped = 0
count_collision_cyc = 0
Scenario_num = 0
newdict = {}

vehicle_ind = tf.where(parsed['state/type']==1)
pedestrian_ind = tf.where(parsed['state/type']==2)
cyclist_ind = tf.where((parsed['state/type']==3))

TTC = 1
SAMPLING_FQ = 10

def data_preprocessing():
# TODO: realized in rect_object.py
    pass


for vehicle in track(vehicle_ind):
    vehicle_rect = rect_object(
        {
        'id':tf.gather(parsed['state/id'],vehicle), 
        'type':tf.gather(parsed['state/type'],vehicle),
        'x':tf.gather(tf.concat([parsed['state/past/x'],parsed['state/current/x'],parsed['state/future/x']],1),vehicle),
        'y':tf.gather(tf.concat([parsed['state/past/y'],parsed['state/current/y'],parsed['state/future/y']],1),vehicle),
        'bbox_yaw':tf.gather(tf.concat([parsed['state/past/bbox_yaw'],parsed['state/current/bbox_yaw'],parsed['state/future/bbox_yaw']],1),vehicle),
        'length':tf.gather(tf.concat([parsed['state/past/length'],parsed['state/current/length'],parsed['state/future/length']],1),vehicle),
        'width':tf.gather(tf.concat([parsed['state/past/width'],parsed['state/current/width'],parsed['state/future/width']],1),vehicle),
        'vel_yaw':tf.gather(tf.concat([parsed['state/past/vel_yaw'],parsed['state/current/vel_yaw'],parsed['state/future/vel_yaw']],1),vehicle),
        'velocity_x':tf.gather(tf.concat([parsed['state/past/velocity_x'],parsed['state/current/velocity_x'],parsed['state/future/velocity_x']],1),vehicle),
        'velocity_y':tf.gather(tf.concat([parsed['state/past/velocity_y'],parsed['state/current/velocity_y'],parsed['state/future/velocity_y']],1),vehicle),
        'validity':tf.gather(tf.concat([parsed['state/past/valid'],parsed['state/current/valid'],parsed['state/future/valid']],1),vehicle)
        }
    )
    for pedestrian in pedestrian_ind:
        pedestrian_rect = rect_object(
        {
        'id':tf.gather(parsed['state/id'],pedestrian), 
        'type':tf.gather(parsed['state/type'],pedestrian),
        'x':tf.gather(tf.concat([parsed['state/past/x'],parsed['state/current/x'],parsed['state/future/x']],1),pedestrian),
        'y':tf.gather(tf.concat([parsed['state/past/y'],parsed['state/current/y'],parsed['state/future/y']],1),pedestrian),
        'bbox_yaw':tf.gather(tf.concat([parsed['state/past/bbox_yaw'],parsed['state/current/bbox_yaw'],parsed['state/future/bbox_yaw']],1),pedestrian),
        'length':tf.gather(tf.concat([parsed['state/past/length'],parsed['state/current/length'],parsed['state/future/length']],1),pedestrian),
        'width':tf.gather(tf.concat([parsed['state/past/width'],parsed['state/current/width'],parsed['state/future/width']],1),pedestrian),
        'vel_yaw':tf.gather(tf.concat([parsed['state/past/vel_yaw'],parsed['state/current/vel_yaw'],parsed['state/future/vel_yaw']],1),pedestrian),
        'velocity_x':tf.gather(tf.concat([parsed['state/past/velocity_x'],parsed['state/current/velocity_x'],parsed['state/future/velocity_x']],1),pedestrian),
        'velocity_y':tf.gather(tf.concat([parsed['state/past/velocity_y'],parsed['state/current/velocity_y'],parsed['state/future/velocity_y']],1),pedestrian),
        'validity':tf.gather(tf.concat([parsed['state/past/valid'],parsed['state/current/valid'],parsed['state/future/valid']],1),pedestrian)
        }
        )

        # pedestrian validity [1,91]
        
        # relation of vehicle and pedstrian [91,], 1 for valid, others not 
        relation_validity = tf.where(tf.squeeze(vehicle_rect.mask & pedestrian_rect.mask)==1,x=True,y=False)
        # print(relation_validity)
        

        result = rect_interaction(vehicle_rect,pedestrian_rect)

        # [91,]
        relation = result.rect_relation(ttc=TTC,sampling_fq=SAMPLING_FQ)
        relation_ = relation & relation_validity
        collised_step = tf.where(relation_==True)
        
        if len(collised_step):
            count_collision_ped+=1
            Scenario_num += 1
            newdict[f"Scenario_{Scenario_num}"]={
                    "Data_record":FILE,
                    "Vehicle_id": str(vehicle_rect.id.numpy()[0]),
                    "VRU_id":str(pedestrian_rect.id.numpy()[0]),
                    "VRU_type":"2",
                    "Start_step":str(collised_step[0,-1].numpy()+1),
                    "End_step":str(collised_step[-1,-1].numpy()+1),
                    "Scenario_type":"1"
                }
        del pedestrian_rect


    for cyclist in cyclist_ind:
        cyclist_rect = rect_object(
        {
        'id':tf.gather(parsed['state/id'],cyclist), 
        'type':tf.gather(parsed['state/type'],cyclist),
        'x':tf.gather(tf.concat([parsed['state/past/x'],parsed['state/current/x'],parsed['state/future/x']],1),cyclist),
        'y':tf.gather(tf.concat([parsed['state/past/y'],parsed['state/current/y'],parsed['state/future/y']],1),cyclist),
        'z':tf.gather(tf.concat([parsed['state/past/z'],parsed['state/current/z'],parsed['state/future/z']],1),cyclist),
        'bbox_yaw':tf.gather(tf.concat([parsed['state/past/bbox_yaw'],parsed['state/current/bbox_yaw'],parsed['state/future/bbox_yaw']],1),cyclist),
        'length':tf.gather(tf.concat([parsed['state/past/length'],parsed['state/current/length'],parsed['state/future/length']],1),cyclist),
        'width':tf.gather(tf.concat([parsed['state/past/width'],parsed['state/current/width'],parsed['state/future/width']],1),cyclist),
        'height':tf.gather(tf.concat([parsed['state/past/height'],parsed['state/current/height'],parsed['state/future/height']],1),cyclist),
        'vel_yaw':tf.gather(tf.concat([parsed['state/past/vel_yaw'],parsed['state/current/vel_yaw'],parsed['state/future/vel_yaw']],1),cyclist),
        'velocity_x':tf.gather(tf.concat([parsed['state/past/velocity_x'],parsed['state/current/velocity_x'],parsed['state/future/velocity_x']],1),cyclist),
        'velocity_y':tf.gather(tf.concat([parsed['state/past/velocity_y'],parsed['state/current/velocity_y'],parsed['state/future/velocity_y']],1),cyclist),
        'validity': tf.gather(tf.concat([parsed['state/past/valid'],parsed['state/current/valid'],parsed['state/future/valid']],1),cyclist)
        }
        )
        # relation of vehicle and pedstrian [91,], 1 for valid, others not 
        relation_validity = tf.where(tf.squeeze(vehicle_rect.mask & cyclist_rect.mask)==1,x=True,y=False)

        result = rect_interaction(vehicle_rect,cyclist_rect)
        relation = result.rect_relation(ttc=TTC,sampling_fq=SAMPLING_FQ)
        relation = relation & relation_validity 

        collised_step = tf.where(relation==True)
        if len(collised_step):
            # print(collised_step)
            # break
            count_collision_cyc+=1
            Scenario_num += 1
            newdict[f"Scenario_{Scenario_num}"]={
                    "Data_record":FILE,
                    "Vehicle_id": str(vehicle_rect.id.numpy()[0]),
                    "VRU_id":str(cyclist_rect.id.numpy()[0]),
                    "VRU_type":"3",
                    "Start_step":str(collised_step[0,-1].numpy()+1),
                    "End_step":str(collised_step[-1,-1].numpy()+1),
                    "Scenario_type":"1"
                }
        del cyclist_rect
            # print(f'vehicle_id:{vehicle_rect.id} and cyclist_id:{cyclist_rect.id} collised.')
            # print(f'Collision start at step {collised_step[0,-1]}, stop at step {collised_step[-1,-1]}')
    del vehicle_rect

END = time.time()
newdict["Statistics"]={
    "Num_vehicle":len(vehicle_ind),
    "Num_pedestrian":len(pedestrian_ind),
    "Num_cyclist":len(cyclist_ind),
    "Count_collision_ped":f"{count_collision_ped}/{len(vehicle_ind)*len(pedestrian_ind)}",
    "Count_collision_cyclist":f"{count_collision_cyc}/{len(vehicle_ind)*len(cyclist_ind)}",
    "Running_time":(END-START)/60
}
newdict["Settings"] = {
    "TTC":TTC,
    "Sampling Frequency": SAMPLING_FQ
}

FILETIME = time.strftime("%Y-%m-%d_%H:%M",time.localtime())
RESULT_FILE = f'result_data={FILE}_{FILETIME}.json'

with open(os.path.join(RESULTDIR,RESULT_FILE),"w") as f:
    json.dump(newdict,f)

del newdict



