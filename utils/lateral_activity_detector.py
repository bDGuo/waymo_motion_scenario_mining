from data_preprocessing import univariate_spline
from math import pi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from rect_object import rect_object
import numpy as np

def lat_act_detector(rect:rect_object,t_s:float,threshold:float,integration_threshold:float,k,smoothing_factor=None)->tuple:
    """
    Determine the lateral activity of the input actor.
    Method:
    1. calculate angle difference
    2. denoising using cubic spline
    3. default lateral activity is all going straight
    4. annotate the turning begin time and end time. 
    ------------------------------------------------------
    Input:
    state:                  rect_objects(data preprocessed)            object
    t_s:                    sample time (second),default=0.1 10hz      float
                            this is aligned with sampling frequency
                            of the dataset
    threshold:              threshold for yaw rate                     float
    integration_threshold:  threshold for yaw angle change             float
    k:                      cubic spline parameter                     int
    smoothing_factor:       smoothing factor for cubic spline          float
    ------------------------------------------------------
    Output:
    la_event: lateral event of sample time          np.array [time_steps=91,]
                0       going straight
                1       turning left
                -1      turning right
                -5      invalid data
    bbox_yaw_rate: cubic splined yaw rate           np.array [time_steps=91,]
    """
    bbox_yaw = rect.kinematics["bbox_yaw"].numpy().squeeze() #[time_steps,]
    la_act = np.zeros_like(bbox_yaw)
    bbox_yaw_rate = np.zeros_like(bbox_yaw)

    valid = tf.where(tf.squeeze(rect.validity)==1).numpy().squeeze() #[valid time_steps,]

    # fast return np.nan if only one bbox_yaw is valid
    if len(valid)<=1:
        for i in range(len(la_act)):
            la_act[i] = np.nan
            bbox_yaw_rate[i] = np.nan
        return la_act,bbox_yaw_rate
        
    la_act[:valid[0]+1] = -5
    la_act[valid[-1]+1:] = -5
    bbox_yaw_rate[:valid[0]] = np.nan
    bbox_yaw_rate[valid[-1]+1:] = np.nan
    # compute yaw rate [valid_length]
    bbox_yaw_valid = bbox_yaw[valid[0]:valid[-1]+1].copy()
    bbox_yaw_valid_rate = __compute_yaw_rate(bbox_yaw_valid) / t_s
    bbox_yaw_rate[valid[0]:valid[-1]+1] = bbox_yaw_valid_rate.copy()
    bbox_yaw_rate,knots = univariate_spline(bbox_yaw_rate,valid,k,smoothing_factor)
    bbox_yaw_valid_rate = bbox_yaw_rate[valid[0]:valid[-1]+1].copy()

    la_act_valid = la_act[valid[0]+1:valid[-1]+1].copy()
    
    # experiment with or with out sliding average.
    # A sliding average may block out very quick turning, which is vital for safety assessment.
 
    iter_yaw_valid_rate = enumerate(bbox_yaw_valid_rate)
    for i,yaw_rate in iter_yaw_valid_rate:
        # too small yaw_rate is taken as going straight
        if np.abs(yaw_rate) <= threshold:
            continue
        # counter clock-wise is turning left and the yaw_rate is positive
        yaw_rate_dir = np.sign(yaw_rate) # 1 for left, -1 for right
        k_end = end_lateral_activity(bbox_yaw_valid_rate[i:],threshold,yaw_rate_dir,integration_threshold,t_s)
        la_act_valid[i:i+k_end] = yaw_rate_dir
        if k_end:
            [next(iter_yaw_valid_rate, None) for _ in range(k_end)]

    la_act[valid[0]+1:valid[-1]+1] = la_act_valid.copy()
    bbox_yaw_rate[valid[0]:valid[-1]+1] = bbox_yaw_valid_rate.copy()
    # assume the first valid lateral activity is same as the second one
    la_act[valid[0]] = la_act_valid[0]
    bbox_yaw_rate[:valid[0]] = -5
    bbox_yaw_rate[valid[-1]+1:] = -5

    return la_act, bbox_yaw_rate

def end_lateral_activity(future_yaw_valid_rate,threshold,current_yaw_dir,integration_threshold,t_s):
    # compute distance from current to the one that is not in the same direction
    integration_yaw_rate = 0
    for i,yaw_rate in enumerate(future_yaw_valid_rate):
        if np.abs(yaw_rate) <= threshold:
            integration_yaw_rate = np.sum(future_yaw_valid_rate[:i]) * t_s * current_yaw_dir
            if integration_yaw_rate >= integration_threshold:
               return i
        if yaw_rate*current_yaw_dir < 0:
            integration_yaw_rate = np.sum(future_yaw_valid_rate[:i]) * t_s * current_yaw_dir

            if integration_yaw_rate >= integration_threshold:
                return i

    if np.sum(future_yaw_valid_rate)*t_s* current_yaw_dir >= integration_threshold:
        return len(future_yaw_valid_rate)
    else:
        return 0

def __compute_yaw_rate(bbox_yaw_valid):
    """
    1. rotate current yaw angle to the previous one
    """
    bbox_yaw_rate_valid = np.zeros_like(bbox_yaw_valid)
    for i in range(1,len(bbox_yaw_valid)):
        x,y = np.cos(bbox_yaw_valid[i]),np.sin(bbox_yaw_valid[i])
        x_ = np.cos(bbox_yaw_valid[i-1])*x + np.sin(bbox_yaw_valid[i-1])*y
        y_ = -np.sin(bbox_yaw_valid[i-1])*x + np.cos(bbox_yaw_valid[i-1])*y
        bbox_yaw_rate_valid[i] = np.arctan2(y_,x_)
    return bbox_yaw_rate_valid





        
