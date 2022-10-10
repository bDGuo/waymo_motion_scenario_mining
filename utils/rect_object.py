# Function to identify whether two rectangulars interacted 
# from cmath import cos
# from selectors import EpollSelector
from abc import ABC
from cmath import nan
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
from math import pi

class rect_object(ABC):
    
    def __init__(self,state):
        self.id = state['id']
        self.type = state['type']
        # [num_object:1,num_timestep:91]
        self.kinematics = {
            'x' : state['x'],
            'y' : state['y'],
        'bbox_yaw': state['bbox_yaw'],
        'length' : state['length'],
        'width' : state['width'],
        'vel_yaw' : state['vel_yaw'], # The yaw angle of each object's velocity vector at each time step.[google]
        'velocity_x' : state['velocity_x'],
        'velocity_y' : state['velocity_y'],
        }
        self.tag = {}
        self.validity = state['validity']

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = - tf.sin(tf.cast(theta, tf.float32)) * x + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)

    def data_preprocessing(self,interp:bool=True,spline:bool=False,moving_average:bool=False):
        # interpolating the invalid data
        # find where data is invalid
        mask = np.where(self.validity.numpy().squeeze()!=1)[0] # [91,]
        valid = np.where(self.validity.numpy().squeeze()==1)[0] # [91,]
        valid_length = len(valid[:])
        appearance_start = valid[0]
        appearance_end = valid[-1]
        appearance_length = appearance_end-appearance_start+1
        if not len(mask):
            return 0
        validity_proportion = valid_length / appearance_length

        # TODO:modify for running in batch in case there is too many invalid time steps.
        # Then this object should be skipped.
        assert validity_proportion > 0.5, f"Valid data proportion too small.Valid/total={validity_proportion:.2f}<50%."
        print(f"Valid/total={validity_proportion:.2f}.")
        # size yaw angle in (0,2*pi) counter clockwise
        self.kinematics['bbox_yaw'] = (self.kinematics['bbox_yaw']+100*pi) % (2*pi)
        self.kinematics['vel_yaw'] = (self.kinematics['vel_yaw']+100*pi) % (2*pi)

        if interp:
            for key in self.kinematics:
                self.kinematics[key] = self.__interpolation(self.kinematics[key],mask,valid)
        if spline:
            for key in self.kinematics:
                self.kinematics[key] = self.__Univariate_spline(self.kinematics[key],mask,valid)
        if moving_average:
            pass
        else:
            pass


    def clean_abnormal_velocity(self,data,valid,t_s,max_acc:float=0.7):
        # return [time_steps,]
        temp = data.squeeze()[valid[0]:valid[-1]+1]
        temp_shift = np.insert(temp[:-1],0,0)
        normal_temp = temp.copy()
        abnormal_temp_indice = np.where(np.abs(temp-temp_shift)>(max_acc*t_s))[0][1:]
        for i in abnormal_temp_indice:
            normal_temp[i] = np.sign(temp[i] - temp[i-1]) *max_acc*t_s + normal_temp[i-1]
        data[valid[0]:valid[-1]+1] = normal_temp
        return data

    def __Univariate_spline(self,data,mask,valid):
        """
        Univariate spline for the noisy data
        ---------------------------------
        Output:[time_steps,]
        """
        temp = data.numpy().astype(np.float32)
        temp = temp.squeeze()
        univariate_spliner = UnivariateSpline(valid,temp[valid])
        time_x = np.arange(len(temp))
        result = univariate_spliner(time_x)
        result[:valid[0]] = np.nan
        result[valid[-1]+1:] = np.nan
        return tf.convert_to_tensor(result,dtype=tf.float32)

    def __simple_moving_average(self,data,mask,valid,kernel_length):
        filtered_data = np.convolve(data[valid[0]:valid[-1]+1],np.ones(kernel_length))\
            [:(valid[-1]-valid[0]+1)] / kernel_length
        # bias correction
        sum_start = 0
        sum_end = 0
        for i in range(kernel_length-1):
            sum_start += filtered_data[i]
            filtered_data[i] = sum_start/ (i+1)
            sum_end += filtered_data[-i]
            filtered_data[-i] = sum_end / (i+1)
        data[valid[0]:valid[-1]+1] = filtered_data.copy()
        data[0,:valid[0]] = np.nan
        data[0,valid[-1]+1:] = np.nan
        return tf.convert_to_tensor(data,dtype=tf.float32)

    def __interpolation(self,data,mask,valid,VELOCITY:bool=False):
        result = data.numpy().astype(np.float32)
        result = result.squeeze()
        result[mask] = np.nan
        temp_pd = pd.DataFrame(result[valid[0]:valid[-1]+1])
        # by default [np.nan,np.nan,1,np.nan,3,np.nan]-> 
                # [np.nan,np.nan,1,2,3,3]
        temp = temp_pd.interpolate().values.T.squeeze()
        result[valid[0]:valid[-1]+1] = temp

        return tf.convert_to_tensor(result,dtype=tf.float32)

class rect_interaction(ABC):
    
    def __init__(self,rect1, rect2):
        '''
        cx,cy: cordinates of center
        '''
        self.r1={'cx':rect1.kinematics['x'],'cy':rect1.kinematics['y'],
                'l':rect1.kinematics['length'], 'w': rect1.kinematics['width'],
                'theta': rect1.kinematics['bbox_yaw'],
                'v_yaw': rect1.kinematics['vel_yaw'],
                'v_x': rect1.kinematics['velocity_x'], 
                'v_y':rect1.kinematics['velocity_y']} #[num_object=1,num_steps]
        
        self.r2={'cx':rect2.kinematics['x'],'cy':rect2.kinematics['y'],
                'l':rect2.kinematics['length'], 'w': rect2.kinematics['width'],
                'theta': rect2.kinematics['bbox_yaw'],
                'v_yaw': rect2.kinematics['vel_yaw'],
                'v_x': rect2.kinematics['velocity_x'], 
                'v_y':rect2.kinematics['velocity_y']} #[num_object=1,num_steps]


    def rect_relation(self,ttc=3.0,sampling_fq=2):
        '''
        two intersected rectangulars             --->    1
        ttc: time-to-collision in seconds, default=3s, with 2 Hz sampling frequency
        else                                     --->    0
        ----------------------------------------------------------------------------
        VAR:
        
        '''
        # Augmenting center cordinates and heading angle with last sampled v_x, v_y and v_yaw
        if ttc * sampling_fq == 0:
            
            r1 = self.r1
            r2 = self.r2
            return self.rect_intersection(r1,r2) | self.rect_intersection(r2,r1)
        else:
            r1 = self.r1
            r2 = self.r2
            r1_,r2_ = self.ttc_esti_center(r1,r2,ttc,sampling_fq)
            relation = (self.rect_intersection(r1_,r2_) | self.rect_intersection(r2_,r1_))
            return relation

            
    def ttc_esti_center(self,r1,r2,ttc=3.0,sampling_fq=2):


        ttc_cx,ttc_cy,ttc_theta = r1['cx'],r1['cy'],r1['theta']
        r1_l,r1_w = r1['l'], r1['w']

        for i in range(1,ttc*sampling_fq+1):
            ttc_cx = tf.concat([ttc_cx,r1['cx']+i/sampling_fq*r1['v_x']],0)
            ttc_cy = tf.concat([ttc_cy,r1['cy']+i/sampling_fq*r1['v_y']],0)
            ttc_theta = tf.concat([ttc_theta,r1['theta']+i/sampling_fq*r1['v_yaw']],0)
            r1['l'] = tf.concat([r1['l'],r1_l],0)
            r1['w'] = tf.concat([r1['w'],r1_w],0)
        r1['cx'] = ttc_cx
        r1['cy'] = ttc_cy
        r1['theta'] = ttc_theta

        ttc_cx,ttc_cy,ttc_theta = r2['cx'],r2['cy'],r2['theta']
        r2_l,r2_w = r2['l'], r2['w']
        for i in range(1,ttc*sampling_fq+1):
            ttc_cx = tf.concat([ttc_cx,r2['cx']+i/sampling_fq*r2['v_x']],0)
            ttc_cy = tf.concat([ttc_cy,r2['cy']+i/sampling_fq*r2['v_y']],0)
            ttc_theta = tf.concat([ttc_theta,r2['theta']+i/sampling_fq*r2['v_yaw']],0)
            r2['l'] = tf.concat([r2['l'],r2_l],0)
            r2['w'] = tf.concat([r2['w'],r2_w],0)
        r2['cx'] = ttc_cx
        r2['cy'] = ttc_cy
        r2['theta'] = ttc_theta

        return r1, r2


    def rect_intersection(self,r1,r2,ind=0) -> bool:
        '''
        if anyone of the four vertices of r2 fall in r1, they are intersected
        c1...c4x(y) are the cordinates of four vertices
        TODO:crossing test https://blog.csdn.net/s0rose/article/details/78831570
        ''' 
        # rotate r1 and r2 cordinate with heading angle of r1
        (r1_cx, r1_cy) = self.cordinate_rotate(r1['cx'],r1['cy'],r1['theta'])
        (r2_cx, r2_cy) = self.cordinate_rotate(r2['cx'],r2['cy'],r1['theta'])
        r2_theta = r2['theta'] - r1['theta']

        r1_l,r1_w = r1['l'], r1['w']
        r2_l,r2_w = r2['l'], r2['w']

        (r2_c1x,r2_c1y) = self.cordinate_rotate(r2_l/2, -r2_w/2,-r2_theta)
        (r2_c2x,r2_c2y) = self.cordinate_rotate(r2_l/2, r2_w/2,-r2_theta) 
        (r2_c3x,r2_c3y) = self.cordinate_rotate(-r2_l/2, -r2_w/2,-r2_theta) 
        (r2_c4x,r2_c4y) = self.cordinate_rotate(-r2_l/2, r2_w/2,-r2_theta) 
        r2_c1x += r2_cx
        r2_c1y += r2_cy
        r2_c2x += r2_cx
        r2_c2y += r2_cy
        r2_c3x += r2_cx
        r2_c3y += r2_cy
        r2_c4x += r2_cx
        r2_c4y += r2_cy

        
        # ll : lower limit, hl: higher limit
        x_ll = r1_cx - r1_l/2
        x_hl = r1_cx + r1_l/2
        y_ll = r1_cy - r1_w/2
        y_hl = r1_cy + r1_w/2

        c1 = tf.less_equal(x_ll,r2_c1x) & tf.less_equal(r2_c1x,x_hl) \
            & tf.less_equal(y_ll,r2_c1y) & tf.less_equal(r2_c1y,y_hl)
        c2 = tf.less_equal(x_ll,r2_c2x) & tf.less_equal(r2_c2x,x_hl) \
            & tf.less_equal(y_ll,r2_c2y) & tf.less_equal(r2_c2y,y_hl)
        c3 = tf.less_equal(x_ll,r2_c3x) & tf.less_equal(r2_c3x,x_hl) \
            & tf.less_equal(y_ll,r2_c3y) & tf.less_equal(r2_c3y,y_hl)
        c4 = tf.less_equal(x_ll,r2_c4x) & tf.less_equal(r2_c4x,x_hl) \
            & tf.less_equal(y_ll,r2_c4y) & tf.less_equal(r2_c4y,y_hl)
        
        c = np.sum((c1 | c2 | c3 | c4).numpy(),0)
        c = tf.convert_to_tensor(c,dtype=tf.bool)

        return c

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = - tf.sin(tf.cast(theta, tf.float32)) * x + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)

