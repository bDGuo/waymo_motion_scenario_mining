import os

from data_preprocessing import univariate_spline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from actor import Actor
import numpy as np
from typing import Tuple
from parameters.tags_dict import la_act_dict

class LatActDetector:
    """
    Detect the lateral activity of the input actor.
    """
    def __init__(self):
        self.lat_act_dict = la_act_dict
        self.new_tag_dict = dict([(v,k) for k,v in self.lat_act_dict.items()])
    def __repr__(self) -> str:
        return f"Lateral Activity Detector: Tags {self.lat_act_dict}."
    
    def tagging(self,rect:Actor,t_s:float,threshold:float,intgr_threshold_turn:float,intgr_threshold_swerv:float,k:int,smoothing_factor=None)->Tuple:
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
        t_s:                    sample time (second),default=0.1->10hz      float
                                this is aligned with sampling frequency
                                of the dataset
        threshold:              threshold for yaw rate                     float
        integration_threshold:  threshold for yaw angle change             float
        k:                      order of splining default:cubic            int
        smoothing_factor:       smoothing factor for cubic spline          float
        ------------------------------------------------------
        Output:
        la_event: lateral event of sample time          np.array [time_steps=91,]
                    0       going straight
                    1       turning left
                    2       swerving left
                    -1      turning right
                    -2      swerving right
                    -5      invalid data
        bbox_yaw_rate: cubic splined yaw rate           np.array [time_steps=91,]
        """
        bbox_yaw = rect.kinematics["bbox_yaw"] #[time_steps,]
        la_act = np.zeros_like(bbox_yaw)
        bbox_yaw_rate = np.zeros_like(bbox_yaw)

        valid = np.where((rect.validity)==1)[0] #[time_steps,]

        # shortcut: return np.nan if only one bbox_yaw is valid
        if len(valid)<=1:
            for i in range(len(la_act)):
                la_act[i] = np.nan
                bbox_yaw_rate[i] = np.nan
            return la_act,bbox_yaw_rate
            
        la_act[:valid[0]+1] = float(self.new_tag_dict['invalid'])
        la_act[valid[-1]+1:] = float(self.new_tag_dict['invalid'])
        bbox_yaw_rate[:valid[0]] = np.nan
        bbox_yaw_rate[valid[-1]+1:] = np.nan
        # compute yaw rate [valid_length]
        bbox_yaw_valid = bbox_yaw[valid[0]:valid[-1]+1].copy()
        bbox_yaw_valid_rate = self.__compute_yaw_rate(bbox_yaw_valid,t_s)
        bbox_yaw_rate[valid[0]:valid[-1]+1] = bbox_yaw_valid_rate.copy()
        # bbox_yaw_rate,knots = univariate_spline(bbox_yaw_rate,valid,k,smoothing_factor)
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
            yaw_rate_dir = np.sign(yaw_rate) # 1 for left, -1 for right, 0 for straight

            t_end,value = self.__end_lateral_activity(bbox_yaw_valid_rate[i:],threshold,yaw_rate_dir,intgr_threshold_turn,intgr_threshold_swerv,t_s,i)
            la_act_valid[i:i+t_end] = yaw_rate_dir*value
            if t_end:
                [next(iter_yaw_valid_rate, None) for _ in range(t_end)]

        la_act[valid[0]+1:valid[-1]+1] = la_act_valid.copy()
        bbox_yaw_rate[valid[0]:valid[-1]+1] = bbox_yaw_valid_rate.copy()
        # assume the first valid lateral activity is same as the second one
        la_act[valid[0]] = la_act_valid[0]
        bbox_yaw_rate[:valid[0]] = -5
        bbox_yaw_rate[valid[-1]+1:] = -5

        return la_act.squeeze(), bbox_yaw_rate.squeeze()

    def __end_lateral_activity(self,future_yaw_valid_rate,threshold,current_yaw_dir,intgr_threshold_turn,intgr_threshold_swerv,t_s,i)->tuple:
        # compute distance from current to the one that is not in the same direction
        # output:
        #   k_end: the index of the end of the lateral activity
        #   value: 1 for turn, 2 for swerve
        integration_yaw_rate = 0
        index_small_yaw_rate = np.where(np.abs(future_yaw_valid_rate) <= threshold)[0]
        index_opposite_yaw_rate = np.where(future_yaw_valid_rate*current_yaw_dir <= 0)[0]

        index_nearest_small_yaw_rate = index_small_yaw_rate[0]     if len(index_small_yaw_rate) else 0
        index_nearest_opposite_yaw_rate = index_opposite_yaw_rate[0] if len(index_opposite_yaw_rate) else 0
        index_end_la_act = min(index_nearest_small_yaw_rate,index_nearest_opposite_yaw_rate)
        
        # if the nearest t_end is the last sample
        if index_end_la_act == 0:
            if np.sum(future_yaw_valid_rate)*t_s* current_yaw_dir >= intgr_threshold_turn:
                return len(future_yaw_valid_rate),1
            elif np.sum(future_yaw_valid_rate)*t_s* current_yaw_dir >= intgr_threshold_swerv:
                return len(future_yaw_valid_rate),2
            else:
                return 0,1
        # nearest t_end is not the last sample
        else:
            integration_t_c_t_end = np.sum(future_yaw_valid_rate[:index_end_la_act]) * t_s * current_yaw_dir
            if integration_t_c_t_end >= intgr_threshold_turn:
                return index_end_la_act,1
            elif integration_t_c_t_end >= intgr_threshold_swerv:
                return index_end_la_act,2
            else:
                return 0,1
            

        # integration_nearest_small_yaw_rate = np.sum(future_yaw_valid_rate[:index_nearest_small_yaw_rate]) * t_s * current_yaw_dir
        # integration_nearest_opposite_yaw_rate = np.sum(future_yaw_valid_rate[:index_nearest_opposite_yaw_rate]) * t_s * current_yaw_dir


        # if integration_nearest_small_yaw_rate >= intgr_threshold_turn:
        #     return index_nearest_small_yaw_rate,1
        # if integration_nearest_opposite_yaw_rate >= intgr_threshold_turn:
        #     return index_nearest_opposite_yaw_rate,1
        # if integration_nearest_opposite_yaw_rate >= intgr_threshold_swerv:
        #     return index_nearest_opposite_yaw_rate,2



    def __compute_yaw_rate(self,bbox_yaw_valid, t_s):
        """
        project bbox heading angle to [-pi,pi]
        """
        bbox_yaw_valid = np.arctan2(np.sin(bbox_yaw_valid),np.cos(bbox_yaw_valid))
        bbox_yaw_rate_valid = bbox_yaw_valid - np.insert(bbox_yaw_valid[:-1],0,0)
        bbox_yaw_rate_valid[0] = 0
        bbox_yaw_rate_valid = np.where(np.abs(bbox_yaw_rate_valid)>=np.pi,\
                                bbox_yaw_rate_valid-np.sign(bbox_yaw_rate_valid)*2*np.pi,\
                                bbox_yaw_rate_valid)
        return bbox_yaw_rate_valid / t_s





            
