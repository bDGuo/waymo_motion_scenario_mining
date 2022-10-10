
import numpy as np
from scipy.interpolate import UnivariateSpline

def clean_abnormal_data(data,valid,t_s,max_acc:float=0.7):
    # return [time_steps,]
    temp = data.squeeze()[valid[0]:valid[-1]+1]
    temp_shift = np.insert(temp[:-1],0,0)
    normal_temp = temp.copy()
    abnormal_temp_indice = np.where(np.abs(temp-temp_shift)>(max_acc*t_s))[0][1:]
    for i in abnormal_temp_indice:
        normal_temp[i] = np.sign(temp[i] - temp[i-1]) *max_acc*t_s + normal_temp[i-1]
    data[valid[0]:valid[-1]+1] = normal_temp
    return data

def univariate_spline(data,valid,k):
    """
    Univariate spline for the noisy data
    ---------------------------------
    Output:[time_steps,]
    """
    temp = data.squeeze()
    if len(temp[valid]) <=5 :
        return temp
    univariate_spliner = UnivariateSpline(valid,temp[valid],k=k)

    time_x = np.arange(len(temp))
    result = univariate_spliner(time_x)
    result[:valid[0]] = np.nan
    result[valid[-1]+1:] = np.nan
    return result

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