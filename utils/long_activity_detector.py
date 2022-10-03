from cmath import nan
import tensorflow as tf
from rect_object import rect_object
import numpy as np

def long_act_detector(rect:rect_object,k_h,max_acc,t_s=0.1,a_cruise=0.1,delta_v=1,time_steps=91,k_cruise=10)->tf.Tensor:
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
    ------------------------------------------------------
    Output:
    lo_event: longitudinal event of sample time    tf.tensor [1,time_steps=91]
             1      accelerating
            -1      decelerating
             0      cruising
             2      stand still
            -2      reversing
            -5      invalid
    long_v: longitudinal speed                      tf.tensor [1,time_steps=91]
    """
    # sanity check
    assert k_h>1,f"Signal window must be greater than 1."
    
    mask = tf.where(tf.squeeze(rect.validity)!=1).numpy().squeeze() #[time_steps,]
    valid = tf.where(tf.squeeze(rect.validity)==1).numpy().squeeze() #[time_steps,]
    # rotating speed in x and y to longitudinal speed
    # [1,time_steps=91]
    (long_v,_) = rect.cordinate_rotate(rect.velocity_x,rect.velocity_y,rect.bbox_yaw)
    # first assuming a cruising event at every data step 
    long_v = long_v.numpy()

    # correct the abnormal data with max acc/dec = 0.7m/s2
    long_v = clean_abnormal_data(long_v,mask,valid,t_s,max_acc=0.7)

    # denoising with sliding average
    kernel= int(k_h)
    filtered_data = np.convolve(long_v[0,valid[0]:valid[-1]+1],np.ones(kernel))[:len(long_v[0,valid[0]:valid[-1]+1])] / kernel

    #bias correction
    # bias correction
    sum_start = 0
    sum_end = 0
    for i in range(kernel-1):
        sum_start += filtered_data[i]
        filtered_data[i] = sum_start/ (i+1)
        sum_end += filtered_data[-i]
        filtered_data[-i] = sum_end / (i+1)
    del sum_start,sum_end


    long_v[0,valid[0]:valid[-1]+1] = filtered_data.copy()
    long_v[0,:valid[0]] = np.nan
    long_v[0,valid[-1]+1:] = np.nan
    # assert len(long_v)>0
    lo_act = np.zeros_like(long_v)
    lo_act[0,:valid[0]] = np.nan
    lo_act[0,valid[-1]+1:] = np.nan
    # print(valid[-1])

    for i in range(0,time_steps,1):
        if i < valid[0] or i > valid[-1]:
            lo_act[0,i] = np.nan
            continue
        # acceleration check
        acc_bool = acceleration(long_v[:,valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
        if acc_bool:
            lo_event,k_end = end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[:,valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,True)
            # print(f"acc.k_end:{k_end},i:{i}")
            lo_act[0,valid[0]+i:valid[0]+k_end+1] = lo_event
            i = k_end
            continue

        # deceleration check
        dec_bool = deceleration(long_v[:,valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
        if dec_bool:
            lo_event,k_end = end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[:,valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,False)
            # print(f"dec.k_end:{k_end},i:{i}")
            lo_act[0,valid[0]+i:valid[0]+k_end+1] = lo_event
            i = k_end
    
    non_cruise_ind = np.where(lo_act[0,valid[0]:valid[-1]+1]!=0)[0] + valid[0]
    if len(non_cruise_ind):
        # start with 0..  cruise
        if 0 < non_cruise_ind[0] < k_cruise:
            lo_act[0,:non_cruise_ind[0]] = lo_act[0,non_cruise_ind[0]]

        # end with  cruise ..0 
        if 0 < len(lo_act[0,valid[0]:valid[-1]+1])-non_cruise_ind[-1] - valid[0] < k_cruise:
            lo_act[0,valid[0]+non_cruise_ind[-1]:valid[-1]+1] = lo_act[0,valid[0]+non_cruise_ind[-1]]

        for i in range(len(non_cruise_ind)-1):
            # check cruising activities shorter than k_cruise
            if (non_cruise_ind[i+1]-non_cruise_ind[i])>=k_cruise:
                continue
            if (non_cruise_ind[i+1]-non_cruise_ind[i]) == 1:
                continue
            else:
                lo_act = removing_short_cruising_act(lo_act,long_v,i,non_cruise_ind,valid[0])

    cruise_ind = np.where(lo_act[0,:]==0)[0]
    small_v_ind = np.where(np.abs(long_v[0,:])<=0.1)[0]
    lo_act[0,np.intersect1d(cruise_ind,small_v_ind)]=2

    # reversing
    lo_act[0,:] = np.where(long_v[0,:]<-0.1,-2,lo_act)

    return tf.convert_to_tensor(lo_act),long_v

def acceleration(long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
    """
    Idea Reference:
    Real-World Scenario Mining for the Assessment of Automated Vehicles
    Author: Erwin de Gelder et.el.
    """
    # condition 3
    # stop = i+k_h if i+k_h <= time_steps else time_steps
    v_min_future = np.min(long_v[0,i:])
    if v_min_future > long_v[0,i]:
        return True
    # for the first k_h samples only check with near future values

    # condition 2
    start = i-k_h+1 if i-k_h+1>=0 else 0
    v_plus = v_plus_calc(long_v[0,start:i+1])
    if v_plus >= a_cruise * k_h * t_s:
        return True

def deceleration(long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
    """
    Idea Reference:
    Real-World Scenario Mining for the Assessment of Automated Vehicles
    Author: Erwin de Gelder et.el.
    """
    # condition 3
    # for last k_h samples do not check with near future values
    # stop = i+k_h if i+k_h <= time_steps else time_steps
    v_max_future = np.max(long_v[0,i:])
    if v_max_future < long_v[0,i]:
        return True
    # for the first k_h samples only check with near future values
    # condition 2
    start = i-k_h+1 if i-k_h+1>=0 else 0
    v_minus = v_minus_calc(long_v[0,start:i+1])
    if v_minus < -a_cruise * k_h * t_s:
        return True
    
    # # condition 1
    # if long_v[0,i-1]==-1:
    #     return True

    # # condition 4
    # j=0
    # for j in range(i+1,time_steps-k_h,1):
    #     v_minus = v_minus_calc(long_v[0,j:j+k_h],j)
    #     if v_minus > -a_cruise * k_h * t_s:
    #         break
    # k_end = j if j < time_steps else i
    # if -long_v[0,k_end]+long_v[0,i] > delta_v:
    #     return True
    # else:
    #     return False

def end_long_activity(i,valid_end,k_h,long_v,a_cruise,t_s,delta_v,ACC:bool=True):
    j=0
    for j in range(i+1,valid_end-k_h,1):
        if ACC:
            v_plus = v_plus_calc(long_v[0,j:j+k_h])
            if v_plus < a_cruise * k_h * t_s:
                break
        else:
            v_minus = v_minus_calc(long_v[0,j:j+k_h])
            if v_minus > -a_cruise * k_h * t_s:
                break
    k_end = j if j < valid_end-k_h-1 else i
    if np.abs(long_v[0,k_end]-long_v[0,i]) > delta_v:
        if ACC:
            return 1,k_end
        else:
            return -1,k_end
    else:
        if ACC:
            return 1,i
        else:
            return -1,i
    

def removing_short_cruising_act(lo_act,long_v,i,non_cruise_ind,valid_start:np.float):
    """
    TODO:think about input and output

    """
    if lo_act[0,non_cruise_ind[i]] == lo_act[0,non_cruise_ind[i+1]]:
        lo_act[0,(non_cruise_ind[i]+1):(non_cruise_ind[i+1])] = lo_act[0,non_cruise_ind[i]]
    # -1 0...0 1
    if lo_act[0,non_cruise_ind[i]]==-1 and lo_act[0,non_cruise_ind[i+1]]==1:    
        v_min_ind = np.argmin(long_v[0,(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
        lo_act[0,(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_min_ind+1)] = -1
        lo_act[0,(non_cruise_ind[i]+1+v_min_ind+1): (non_cruise_ind[i+1])] = 1
    # 1 0...0 -1
    if lo_act[0,non_cruise_ind[i]]==1 and lo_act[0,non_cruise_ind[i+1]]==-1:  
        v_max_ind = np.argmax(long_v[0,(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
        lo_act[0,(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_max_ind+1)] = 1
        lo_act[0,(non_cruise_ind[i]+1+v_max_ind+1): (non_cruise_ind[i+1])] = -1

    return lo_act


def clean_abnormal_data(data,mask,valid,t_s,max_acc=0.7):
    temp = data[0,valid[0]:valid[-1]+1]
    temp_shift = np.insert(temp[:-1],0,0)
    normal_temp = temp.copy()
    abnormal_temp_indice = np.where(np.abs(temp-temp_shift)>(max_acc*t_s))[0][1:]
    for i in abnormal_temp_indice:
        normal_temp[i] = np.sign(temp[i] - temp[i-1]) *0.7*t_s + normal_temp[i-1]
    data[0,valid[0]:valid[-1]+1] = normal_temp
    return data

def v_plus_calc(v):
    return v[-1] - np.min(v)

def v_minus_calc(v):
    return v[-1] - np.max(v)

    