from ast import iter_child_nodes
from math import pi,atan,tan
import tensorflow as tf
from rect_object import rect_object
import numpy as np

def lat_act_detector(rect:rect_object,k_h:int,kernel:int,threshold:float,integration_threshold:float)->tf.Tensor:
    """
    Determine the lateral activity of the input actor.
    Idea is that if the sum of the difference of actor velocity yaw angle and its bounding box yaw angle
    in a consecutive time over or below a certain threshold, then the turning is detected.
    The threshold is determined by hand.
    As a constructed road such as intersections and urban roads, turning means trajectory 
    change more than 45 degree.
    Method:
    1. calculate angle difference
    2. denoising using simple moving average
    3. default lateral activity is all going straight
    4. annotate the turning begin time and end time. consider it as an acceleration detection
    5. remove short going straight. 
    ------------------------------------------------------
    Input:
    state:      rect_objects(data preprocessed)            object
    k_h:        sample window (step)                       tf.int
    t_s:        sample time (second),default=0.1 10hz      tf.float
                this is aligned with sampling frequency
                of the dataset
    a_cruise:   maximum average acceleration               tf.float
                default= 0.1 m/s^2

    delta_v:    minimm speed increase                     tf.float
                default=1 m/s
    time_steps: num_steps in a state                       tf.int
    k_cruise:   threshold num_steps in case of              int
                very short cruising activities.
    ------------------------------------------------------
    Output:
    la_event: lateral event of sample time    tf.tensor [1,time_steps=91]
            0       going straight
            1       turning left
            -1      turning right
            np.nan  invalid data

    la_v: lateral speed                      tf.tensor [1,time_steps=91]
    """
    # sanity check
    assert k_h>1,f"Signal window must be greater than 1."
    assert kernel>1,f"Signal window must be greater than 1."
    bbox_yaw = rect.bbox_yaw.numpy().squeeze() #[time_steps,]
    la_act = np.zeros_like(bbox_yaw)
    bbox_yaw_rate = np.zeros_like(bbox_yaw)

    valid = tf.where(tf.squeeze(rect.validity)==1).numpy().squeeze() #[valid time_steps,]

    # fast return np.nan if only one bbox_yaw is valid
    # TODO:fix the error in x axis in try.ipynb while plotting.
    if len(valid)<=1:
        for i in range(len(la_act)):
            la_act[i] = np.nan
        return tf.convert_to_tensor(la_act)
        
    la_act[:valid[0]+1] = np.nan
    la_act[valid[-1]+1:] = np.nan
    bbox_yaw_rate[:valid[0]+1] = np.nan
    bbox_yaw_rate[valid[-1]+1:] = np.nan

    # compute yaw rate [valid_length]
    bbox_yaw_valid = bbox_yaw[valid[0]:valid[-1]+1]
    # size yaw angel in (-pi,pi)
    bbox_yaw_valid = np.arctan(np.tan(bbox_yaw_valid))
    bbox_yaw_valid_shift = np.ones_like(bbox_yaw_valid)
    bbox_yaw_valid_shift = np.insert(bbox_yaw_valid[:-1],0,0)
    bbox_yaw_valid_rate = (bbox_yaw_valid_shift - bbox_yaw_valid)[1:]
    la_act_valid = la_act[valid[0]+1:valid[-1]+1].copy()
    
    # denoising with sliding average
    # TODO:experiment with or with out sliding average.
    # A sliding average may block out very quick turning, which is vital for safety assesment.
    # bbox_yaw_valid_rate = sliding_average(int(k_h),bbox_yaw_valid_rate,valid_length,valid)
    iter_yaw_valid_rate = enumerate(bbox_yaw_valid_rate)
    for i,yaw_rate in iter_yaw_valid_rate:
        # too small yaw_rate is taken as going straight
        if np.abs(yaw_rate) <= threshold:
            continue
        # clock-wise is turning left and the yaw_rate is positive
        yaw_rate_dir = np.sign(yaw_rate) # 1 for left, -1 for right
        k_end = end_lateral_activity(bbox_yaw_valid_rate[i:],threshold,yaw_rate_dir,integration_threshold)
        la_act_valid[i:i+k_end] = yaw_rate_dir
        # if i<=15:
        #     print(f"i:{i},k_end:{k_end}")
        if k_end:
            [next(iter_yaw_valid_rate, None) for _ in range(k_end)]
        

    la_act[valid[0]+1:valid[-1]+1] = la_act_valid
    bbox_yaw_rate[valid[0]+1:valid[-1]+1] = bbox_yaw_valid_rate


    return tf.convert_to_tensor(la_act),bbox_yaw_rate

def end_lateral_activity(future_yaw_valid_rate,threshold,current_yaw_dir,integration_threshold):
    # compute distance from current to the one that is not in the same direction
    integration_yaw_rate = 0
    for i,yaw_rate in enumerate(future_yaw_valid_rate):
        if np.abs(yaw_rate) <= threshold:
            integration_yaw_rate = np.sum(future_yaw_valid_rate[:i])

            if integration_yaw_rate >= integration_threshold:
               return i
        if yaw_rate*current_yaw_dir < 0:
            integration_yaw_rate = np.sum(future_yaw_valid_rate[:i])
            if integration_yaw_rate >= integration_threshold:
                return i
    if np.sum(future_yaw_valid_rate) >= integration_threshold:
        return i
    else:
        return 0

def sliding_average(kernel:int,bbox_yaw_valid_rate,valid_length:int,valid):    
    filtered_data = np.convolve(bbox_yaw_valid_rate,np.ones(kernel))[:valid_length] / kernel
    # bias correction
    sum_start = 0
    sum_end = 0
    for i in range(kernel-1):
        sum_start += filtered_data[i]
        filtered_data[i] = sum_start/ (i+1)
        sum_end += filtered_data[-i]
        filtered_data[-i] = sum_end / (i+1)
    return filtered_data



        
