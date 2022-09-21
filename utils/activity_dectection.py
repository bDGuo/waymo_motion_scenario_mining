import tensorflow as tf
from rect_object import rect_object
import numpy as np

def long_act_detector(rect:rect_object,k_h,t_s=0.1,a_cruise=0.1,delta_v=0.1,time_steps=91):
    """
    Determine the longitudial activity of the input actor.
    ------------------------------------------------------
    Input:
    state:      rect_objects
    k_h:        sample window (step)                       tf.int
    t_s:        sample time (second),default=0.1 10hz      tf.float
                this is aligned with sampling frequency
                of the dataset
    a_cruise:   maximum average acceleration               tf.float
                default= 0.1 m/s^2

    delta_v:    minimm speed increase                     tf.float
                default=1 m/s
    time_steps: num_steps in a state                       tf.int
    ------------------------------------------------------
    Output:
    lo_event: longitudinal event of sample time    tf.tensor [1,time_steps=91]
            1 for accelerating
            -1 for decelerating
            0 for cruising
    long_v: longitudinal speed                      tf.tensor [1,time_steps=91]
    
    """

    # interpolation of invalid values
    # [1,time_steps=91]
    rect.data_preprocessing()
    # rotating speed in x and y to longitudinal speed
    # [1,time_steps=91]
    (long_v,_) = rect.cordinate_rotate(rect.velocity_x,rect.velocity_y,rect.bbox_yaw)
    # first assuming a cruising event at every data step 
    # TODO: for the signal window padding head and tail or not?
    long_v = long_v.numpy()
    lo_act = np.zeros_like(long_v)
    for i in range(k_h,time_steps-k_h+1,1):
        # acceleration check
        acc_bool = acceleration(long_v,i,k_h,t_s,a_cruise,delta_v)
        if acc_bool:
            lo_act[0,i] = 1
            break
        # deceleration check
        dec_bool = deceleration(long_v,i,k_h,t_s,a_cruise,delta_v)
        if dec_bool:
            lo_act[0,i] = -1
    
    # TODO:small Cruising check
             
    
    return tf.convert_to_tensor(lo_act)

def acceleration(long_v,i,k_h,t_s,a_cruise,delta_v):
    """
    TODO:complete with 4 conditions
    Idea Reference:
    Real-World Scenario Mining for the Assessment of Automated Vehicles
    Author: Erwin de Gelder et.el.
    """
    # condition 2
    v_plus = v_plus_calc(long_v,i,k_h)
    if v_plus >= a_cruise * k_h * t_s:
        return True
    
    # condition 3
    v_min_future = np.min(long_v[0,i:i+k_h+1])
    if v_min_future >= long_v[0,i]:
        return True
    
    # condition 1
    if long_v[0,i-1]:
        return True

    # condition 4
    # TODO:maybe leave this condition after going through?
    

    





    pass

def deceleration(long_v,i,k_h,t_s,a_cruise,delta_v):
    """
    TODO:complete with 4 conditions
    """
    v_max = tf.math.maximum(long_v[0,(i-k_h):i+1])
    pass

def cruising_check():
    """
    TODO:think about input and output
    """
    pass

def v_plus_calc(v,i,k_h):
    v_min = np.min(v[0,(i-k_h):i+1])
    return v[0,i] - v_min

def v_minus_calc(v,i,k_h):
    v_max = np.max(v[0,(i-k_h):i+1])
    return v[0,i] - v_max