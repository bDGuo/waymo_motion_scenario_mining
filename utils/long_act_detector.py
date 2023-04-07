import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from actor import Actor
import numpy as np
from data_preprocessing import univariate_spline
from logger.logger import logger
from parameters.tags_dict import lo_act_dict

class LongActDetector:
    """
    Detect the lateral activity of the input actor.
    """
    def __init__(self) -> None:
        self.lo_act_dict = lo_act_dict
        self.new_tag_dict = dict([(v,k) for k,v in self.lo_act_dict.items()])

    def __repr__(self) -> str:
        return f"Longitudinal Activity Detector: Tags {self.lo_act_dict}."
        

    def tagging(self,rect:Actor,k_h,max_acc,t_s,a_cruise,delta_v,time_steps,k_cruise,k,smoothing_factor=None)->tuple:
        """
        Determine the longitudial activity of the input actor.
        ------------------------------------------------------
        Input:
        state:      rect_objects
        k_h:        sample window (step)                        int
        t_s:        sample time (second),default=0.1 10hz       float
                    this is aligned with sampling frequency
                    of the dataset
        a_cruise:   maximum average acceleration                float
                    default= 0.1 m/s^2
        delta_v:    minimm speed increase                       float
                    default=1 m/s
        time_steps: num_steps in a state                        int
        k_cruise:   threshold num_steps in case of              int
                    very short cruising activities.
        k:          order for univariate spline                 int
        ------------------------------------------------------
        Output:
        lo_act: longitudinal activity of sample time    np.array [time_steps=91,]
        refer the parameters.tag_dict for the meaning of the tags
        long_v: splined longitudinal speed                 np.array [time_steps=91,]
        long_v1: not splined long. speed
        knots:  #knots of splining
        """
        # sanity check
        if k_h<=1:
            logger.warn(f"Signal window must be greater than 1.")

        valid = np.where((rect.validity)==1)[0] #[time_steps,]
        # rotating speed in x and y to longitudinal speed
        # [1,time_steps=91]
        (long_v1,_) = rect.cordinate_rotate(rect.kinematics['velocity_x'],\
                                        rect.kinematics['velocity_y'],\
                                        rect.kinematics['bbox_yaw'])
        
        long_v,knots = univariate_spline(long_v1,valid,k,smoothing_factor)
        lo_act,long_v = self.__long_act_detector_core(long_v,long_v1,valid,rect,k_h,t_s,a_cruise,time_steps,delta_v,k_cruise)
        return lo_act,long_v,long_v1,knots

    def __long_act_detector_core(self,long_v,long_v1,valid,rect,k_h,t_s,a_cruise,time_steps,delta_v,k_cruise):
        lo_act = np.zeros_like(long_v)
        lo_act[:valid[0]] = float(self.new_tag_dict['invalid'])
        lo_act[valid[-1]+1:] = float(self.new_tag_dict['invalid'])
        for i in range(valid[0],valid[-1]+1):
            # acceleration check
            acc_bool = self.__acceleration(long_v[valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
            if acc_bool:
                lo_event,k_end = self.__end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,True)
                lo_act[i:valid[0]+k_end+1] = lo_event
                # print(f"i:{i},acc:{acc_bool},k_end:{k_end},valid start:{valid[0]}")
                i = k_end
                continue
            # deceleration check
            dec_bool = self.__deceleration(long_v[valid[0]:valid[-1]+1],i-valid[0],k_h,t_s,a_cruise,time_steps,delta_v)
            if dec_bool:
                lo_event,k_end = self.__end_long_activity(i-valid[0],valid[-1]+1-valid[0],k_h,long_v[valid[0]:valid[-1]+1],a_cruise,t_s,delta_v,False)
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
                    lo_act = self.__removing_short_cruising_act(lo_act,long_v,i,non_cruise_ind,valid[0])
        cruise_ind = np.where(lo_act==0)[0]
        small_v_ind = np.where(np.abs(long_v1)*t_s<=0.01*rect.kinematics['length'].squeeze()[valid][-1])[0]
        lo_act[np.intersect1d(cruise_ind,small_v_ind)] = float(self.new_tag_dict['standing still'])
        # reversing
        lo_act = np.where(long_v1*t_s<-0.01*rect.kinematics['length'].squeeze()[valid][-1],float(self.new_tag_dict['reversing']),lo_act)
        long_v[:valid[0]] = -5
        long_v[valid[-1]+1:] = -5
        return lo_act.squeeze(),long_v.squeeze()

    def __acceleration(self,valid_long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
        """
        Idea Reference:
        Real-World Scenario Mining for the Assessment of Automated Vehicles
        Author: Erwin de Gelder et.el.
        """
        # condition 3
        v_min_future = np.min(valid_long_v[i:i+k_h+1])
        # condition 2
        start = i-k_h+1 if i-k_h+1>=0 else 0
        v_plus = self.__v_plus_calc(valid_long_v[start:i+1])

        if v_min_future > valid_long_v[i]:
            return True
        if v_plus >= a_cruise * k_h * t_s:
            return True
        else:
            return False


    def __deceleration(self,valid_long_v,i,k_h,t_s,a_cruise,time_steps,delta_v):
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
        v_minus = self.__v_minus_calc(valid_long_v[start:i+1])
        if v_max_future < valid_long_v[i]:
            return True
        if v_minus <= -a_cruise * k_h * t_s:
            return True
        else:
            return False

    def __end_long_activity(self,i,valid_end,k_h,valid_long_v,a_cruise,t_s,delta_v,ACC:bool=True):
        j=i+1
        for j in range(i+1,valid_end-k_h,1):
            if ACC:
                v_plus = self.__v_plus_calc(valid_long_v[j:j+k_h])
                if v_plus < a_cruise * k_h * t_s:
                    break
            else:
                v_minus = self.__v_minus_calc(valid_long_v[j:j+k_h])
                if v_minus > -a_cruise * k_h * t_s:
                    break
        k_end = j if j < valid_end-k_h-1 else i
        if np.abs(valid_long_v[k_end]-valid_long_v[i]) > delta_v:
            if ACC:
                return float(self.new_tag_dict['accelerating']),k_end
            else:
                return float(self.new_tag_dict['decelerating']),k_end
        else:
            if ACC:
                return float(self.new_tag_dict['accelerating']),i
            else:
                return float(self.new_tag_dict['decelerating']),i
        

    def __removing_short_cruising_act(self,lo_act,long_v,i,non_cruise_ind,valid_start:float):
        """
        remove short cruising activities
        e.g.
        -1 0...0 1 => -1 -1 -1 ... 1 1 1
        vice versa

        """
        if lo_act[non_cruise_ind[i]] == lo_act[non_cruise_ind[i+1]]:
            lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])] = lo_act[non_cruise_ind[i]]
        # -1 0...0 1
        if lo_act[non_cruise_ind[i]]==-1 and lo_act[non_cruise_ind[i+1]]==1:    
            v_min_ind = np.argmin(long_v[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
            lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_min_ind+1)] = float(self.new_tag_dict['decelerating'])
            lo_act[(non_cruise_ind[i]+1+v_min_ind+1): (non_cruise_ind[i+1])] = float(self.new_tag_dict['accelerating'])
        # 1 0...0 -1
        if lo_act[non_cruise_ind[i]]==1 and lo_act[non_cruise_ind[i+1]]==-1:  
            v_max_ind = np.argmax(long_v[(non_cruise_ind[i]+1):(non_cruise_ind[i+1])])
            lo_act[(non_cruise_ind[i]+1):(non_cruise_ind[i]+1+v_max_ind+1)] = float(self.new_tag_dict['accelerating'])
            lo_act[(non_cruise_ind[i]+1+v_max_ind+1): (non_cruise_ind[i+1])] = float(self.new_tag_dict['decelerating'])

        return lo_act

    def __v_plus_calc(self,v):
        return v[-1] - np.min(v)

    def __v_minus_calc(self,v):
        return v[-1] - np.max(v)


