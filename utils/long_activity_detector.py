import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from rect_object import rect_object
import numpy as np
from data_preprocessing import univariate_spline


def long_act_detector(rect:rect_object,k_h,max_acc,t_s=0.1,a_cruise=0.1,delta_v=1,time_steps=91,k_cruise=10,k=3,smoothing_factor=None)->tuple:
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
    k_cruise:   threshold num_steps in case of              int
                very short cruising activities.
    k:          order for univariate spline                 int
    ------------------------------------------------------
    Output:
    lo_event: longitudinal event of sample time    tf.tensor [1,time_steps=91]
             1      accelerating
            -1      decelerating
             0      cruising
             2      stand still
            -2      reversing
            -5      invalid
    long_v: splined longitudinal speed                 tf.tensor [1,time_steps=91]
    long_v1: not splined long. speed
    knots:  #knots of splining
    """
    # sanity check
    assert k_h>1,f"Signal window must be greater than 1."

    valid = tf.where(tf.squeeze(rect.validity)==1).numpy().squeeze() #[time_steps,]
    # rotating speed in x and y to longitudinal speed
    # [1,time_steps=91]
    (long_v1,_) = rect.cordinate_rotate(rect.kinematics['velocity_x'],\
                                    rect.kinematics['velocity_y'],\
                                    rect.kinematics['vel_yaw'])
    
    
    long_v,knots = univariate_spline(long_v1.numpy(),valid,k,smoothing_factor)
    # correct the abnormal data with max acc/dec = 0.7m/s2
    # long_v = clean_abnormal_data(long_v,valid,t_s,max_acc=max_acc)
    # assert len(long_v)>0
    lo_act = np.zeros_like(long_v)
    lo_act[:valid[0]] = -5
    lo_act[valid[-1]+1:] = -5
    # print(valid[-1])

    for i in range(valid[0],valid[-1]+1):
        # acceleration check
        acc_bool = acceleration(long_v[valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
        if acc_bool:
            lo_event,k_end = end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,True)
            lo_act[i:valid[0]+k_end+1] = lo_event
            # print(f"i:{i},acc:{acc_bool},k_end:{k_end},valid start:{valid[0]}")
            i = k_end
            continue

        # deceleration check
        dec_bool = deceleration(long_v[valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
        if dec_bool:
            lo_event,k_end = end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,False)
            lo_act[i:valid[0]+k_end+1] = lo_event
            # print(f"i:{i},dec:{dec_bool},k_end:{k_end},valid start:{valid[0]}")
            i = k_end
    
    non_cruise_ind = np.where(lo_act[valid[0]:valid[-1]+1]!=0)[0] + valid[0]
    # print(non_cruise_ind)
    if len(non_cruise_ind):
        # start with 0..  
        if 0 < non_cruise_ind[0] - valid[0] < k_cruise:
            lo_act[valid[0]:non_cruise_ind[0]] = lo_act[non_cruise_ind[0]]

        # end with  cruise ..0 
        if 0 < valid[-1]-non_cruise_ind[-1]< k_cruise:
            lo_act[non_cruise_ind[-1]:valid[-1]+1] = lo_act[non_cruise_ind[-1]]

        for i in range(len(non_cruise_ind)-1):
            # check cruising activities shorter than k_cruise
            if (non_cruise_ind[i+1]-non_cruise_ind[i])>=k_cruise:
                continue
            if (non_cruise_ind[i+1]-non_cruise_ind[i]) == 1:
                continue
            else:
                lo_act = removing_short_cruising_act(lo_act,long_v,i,non_cruise_ind,valid[0])

    cruise_ind = np.where(lo_act==0)[0]
    small_v_ind = np.where(np.abs(long_v)*t_s<=0.01*rect.kinematics['length'].numpy().squeeze()[valid][-1])[0]
    lo_act[np.intersect1d(cruise_ind,small_v_ind)]=2

    # reversing
    lo_act = np.where(long_v<-0.1,-2,lo_act)
    
    long_v[:valid[0]] = -5
    long_v[valid[-1]+1:] = -5

    return lo_act,long_v,long_v1,knots

def acceleration(valid_long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
    """
    Idea Reference:
    Real-World Scenario Mining for the Assessment of Automated Vehicles
    Author: Erwin de Gelder et.el.
    """
    # condition 3
    v_min_future = np.min(valid_long_v[i:i+k_h+1])
    # condition 2
    start = i-k_h+1 if i-k_h+1>=0 else 0
    v_plus = v_plus_calc(valid_long_v[start:i+1])

    if v_min_future > valid_long_v[i]:
        return True
    if v_plus >= a_cruise * k_h * t_s:
        return True
    else:
        return False


def deceleration(valid_long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
    """
    Idea Reference:
    Real-World Scenario Mining for the Assessment of Automated Vehicles
    Author: Erwin de Gelder et.el.
    """
    # condition 3
    # for last k_h samples do not check with near future values
    # stop = i+k_h if i+k_h <= time_steps else time_steps
    v_max_future = np.max(valid_long_v[i:i+k_h+1])
    start = i-k_h+1 if i-k_h+1>=0 else 0
    v_minus = v_minus_calc(valid_long_v[start:i+1])
    if v_max_future < valid_long_v[i]:
        return True
    if v_minus <= -a_cruise * k_h * t_s:
        return True
    else:
        return False

def end_long_activity(i,valid_end,k_h,valid_long_v,a_cruise,t_s,delta_v,ACC:bool=True):
    j=i+1
    for j in range(i+1,valid_end-k_h,1):
        if ACC:
            v_plus = v_plus_calc(valid_long_v[j:j+k_h])
            if v_plus < a_cruise * k_h * t_s:
                break
        else:
            v_minus = v_minus_calc(valid_long_v[j:j+k_h])
            if v_minus > -a_cruise * k_h * t_s:
                break
    k_end = j if j < valid_end-k_h-1 else i
    if np.abs(valid_long_v[k_end]-valid_long_v[i]) > delta_v:
        if ACC:
            return 1,k_end
        else:
            return -1,k_end
    else:
        if ACC:
            return 1,i
        else:
            return -1,i
    

def removing_short_cruising_act(lo_act,long_v,i,non_cruise_ind,valid_start:float):
    """
    TODO:think about input and output

    """
    if lo_act[non_cruise_ind[i]] == lo_act[non_cruise_ind[i+1]]:
        lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])] = lo_act[non_cruise_ind[i]]
    # -1 0...0 1
    if lo_act[non_cruise_ind[i]]==-1 and lo_act[non_cruise_ind[i+1]]==1:    
        v_min_ind = np.argmin(long_v[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
        lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_min_ind+1)] = -1
        lo_act[(non_cruise_ind[i]+1+v_min_ind+1): (non_cruise_ind[i+1])] = 1
    # 1 0...0 -1
    if lo_act[non_cruise_ind[i]]==1 and lo_act[non_cruise_ind[i+1]]==-1:  
        v_max_ind = np.argmax(long_v[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
        lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_max_ind+1)] = 1
        lo_act[(non_cruise_ind[i]+1+v_max_ind+1): (non_cruise_ind[i+1])] = -1

    return lo_act

def v_plus_calc(v):
    return v[-1] - np.min(v)

def v_minus_calc(v):
    return v[-1] - np.max(v)


