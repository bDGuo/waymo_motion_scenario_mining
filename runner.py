import math
import uuid
import time
import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf
import json
from rich.progress import track
import os
import utils
from utils.data_parser import features_description
from utils.rect_object import *



if __name__=="__main__":
    
    # "global settings"
    ROOT = os.path.abspath(os.path.dirname(""))
    DATADIR = os.path.join(ROOT,"waymo_open_dataset","data","tf_example","training")
    RESULTDIR = os.path.join(ROOT,"results")

    # Create Dataset
    # TODO: change here for read all files 
    FILE = "training_tfexample.tfrecord-00000-of-01000"
    FILENAME = os.path.join(DATADIR,FILE)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, features_description)

    # time measuring. 
    # include this if other function also need time measuring and logging.
    # START = time.time()
    # END = time.time()

    