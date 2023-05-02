# Function to identify whether two rectangular interacted

from abc import ABC
from math import cos, sin

import numpy as np
import pandas as pd
import tensorflow as tf
from shapely.geometry import Polygon
from shapely.ops import unary_union

from logger.logger import *


class Actor(ABC):
    def __init__(self, state: dict):
        self.id = state['id']
        self.type = state['type']
        # [num_object:1,num_timestep:91] np.array
        self.kinematics = {
            'x': state['x'],
            'y': state['y'],
            'bbox_yaw': state['bbox_yaw'],  #
            'length': state['length'],
            'width': state['width'],
            'vel_yaw': state['vel_yaw'],  # The yaw angle of each object's velocity vector at each time step.[google]
            'velocity_x': state['velocity_x'],
            'velocity_y': state['velocity_y']
        }
        self.tag = {}
        self.validity = state['validity']
        self.validity_ratio = 1
        self.expanded_multipolygon = []
        self.expanded_polygon = []
        self.expanded_bboxes = []
        self.time_steps = self.kinematics['x'].shape[0]

    def ts_to_np(self):
        """
        convert input tensor to numpy
        """
        for key in self.kinematics:
            self.kinematics[key] = self.kinematics[key].numpy().squeeze()
        self.validity = self.validity.numpy().squeeze()
        self.id = self.id.numpy().squeeze()
        self.type = self.type.numpy().squeeze()

    def cordinate_rotate_ts(self, x, y, theta):
        '''
        tensor version of cordinate_rotate
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = tf.sin(tf.cast(theta, tf.float32)) * (-x) + tf.cos(tf.cast(theta, tf.float32)) * y
        return x_, y_
    
    def cordinate_rotate(self,x,y,theta):
        '''
        numpy version of cordinate_rotate
        rotate the cordinate system with r1's heading angle as x_ axis
        formula:
        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = np.cos(theta) * x + np.sin(theta) * y
        y_ = np.sin(theta) * (-x) + np.cos(theta) * y
        return x_, y_

    def data_preprocessing(self, interp: bool = True,input_type='np'):
        """
        1. transform the data to numpy
        2. interpolate the invalid data
        return validity/appearance
        """
        if input_type == 'tensor':
            self.ts_to_np()
        mask = np.where(self.validity!= 1)[0]  # [num_time_steps,]
        valid = np.where(self.validity== 1)[0]  # [num_time_steps,]
        valid_length = len(valid)
        appearance_start = valid[0]
        appearance_end = valid[-1]
        appearance_length = appearance_end - appearance_start + 1
        # resize yaw angle in [-pi,pi)
        self.kinematics['bbox_yaw'] = self.__project_angel(self.kinematics['bbox_yaw'])
        self.kinematics['vel_yaw'] = self.__project_angel(self.kinematics['vel_yaw'])
        if len(mask) == 0:
            return 1
        validity_proportion = valid_length / appearance_length
        if validity_proportion < 0.5:
            logger.warning(f"Valid data proportion too small. Valid/total={validity_proportion:.2f}<50%.")
        self.validity_ratio = validity_proportion

        if interp:
            for key in self.kinematics:
                self.kinematics[key] = self.__interpolation(self.kinematics[key], mask, valid)

        return f"{validity_proportion:.2f}"

    @property
    def long_v(self):
        return self.long_v

    @long_v.setter
    def long_v(self, value):
        self.long_v = value

    @property
    def yaw_rate(self):
        return self.yaw_rate

    @yaw_rate.setter
    def yaw_rate(self, value):
        self.yaw_rate = value

    def clean_abnormal_velocity(self, data, valid, t_s, max_acc: float = 0.7):
        # return [time_steps,]
        temp = data.squeeze()[valid[0]:valid[-1] + 1]
        temp_shift = np.insert(temp[:-1], 0, 0)
        normal_temp = temp.copy()
        abnormal_temp_indice = np.where(np.abs(temp - temp_shift) > (max_acc * t_s))[0][1:]
        for i in abnormal_temp_indice:
            normal_temp[i] = np.sign(temp[i] - temp[i - 1]) * max_acc * t_s + normal_temp[i - 1]
        data[valid[0]:valid[-1] + 1] = normal_temp
        return data

    # this method is moved to data_preprocessing.py as a seperate function
    def __univariate_spline(self, data, valid, k=3, smoothing_factor=None):
        ...
    def __simple_moving_average(self, data, mask, valid, kernel_length):
        ...

    def __project_angel(self, angle):
        '''
        project the angle to [-pi,pi)
        '''
        return np.arctan2 (np.sin(angle), np.cos(angle))

    def __interpolation(self, data, mask, valid, VELOCITY: bool = False):
        result = data.astype(np.float32).squeeze()
        result[mask] = np.nan
        temp_pd = pd.DataFrame(result[valid[0]:valid[-1] + 1]).interpolate()
        # by default [np.nan,np.nan,1,np.nan,3,np.nan]-> 
        # [np.nan,np.nan,1,2,3,3]
        temp = temp_pd.values.T.squeeze() if temp_pd is not None else -1
        result[valid[0]:valid[-1] + 1] = temp
        return result.astype(np.float32)

    def instant_polygon(self, x: float, y: float, yaw_angle: float, length: float, width: float):
        """
        actor polygon at every time step
        """
        polygon_coordinates = [
            (length / 2, - width / 2),
            (length / 2,  width / 2),
            (-length / 2, width / 2),
            (-length / 2, -width / 2)
        ]
        for i in range(4):
            new_x,new_y = self.cordinate_rotate(polygon_coordinates[i][0],polygon_coordinates[i][1],-yaw_angle)
            polygon_coordinates[i] = (new_x+x, new_y+y)
        return Polygon(polygon_coordinates)

    def get_validity_range(self):
        valid = np.where(self.validity == 1)[0] # [time_steps,]
        return valid[0], valid[-1]

    def polygon_set(self):
        """
        return a set of polygon at every time step
        """
        polygon_set = []
        x_ = self.kinematics['x']
        y_ = self.kinematics['y']
        yaw_ = self.kinematics['bbox_yaw']
        length_ = self.kinematics['length']
        width_ = self.kinematics['width']
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                polygon_set.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            else:
                polygon_set.append(self.instant_polygon(x_[i], y_[i], yaw_[i], length_[i], width_[i]))
        return polygon_set
    
    def set_yaw_rate(self, yaw_rate):
        self.kinematics['yaw_rate'] = yaw_rate
    
    def get_kinematics(self)->dict:
        return self.kinematics

    def expanded_polygon_set(self, TTC: int, sampling_fq: int, yaw_rate):
        """
        CTRV model: constant turn rate and velocity
        Using CTRV to predict the future trajectory of the ego vehicle in a shorter time(<=3s)
        """
        self.expanded_multipolygon = []
        expanded_polygon_set = []
        x_ = self.kinematics['x']
        y_ = self.kinematics['y']
        yaw_ = self.kinematics['bbox_yaw']
        length_ = self.kinematics['length']
        width_ = self.kinematics['width']
        vx_ = self.kinematics['velocity_x']
        vy_ = self.kinematics['velocity_y']
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                expanded_polygon_set.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
                self.expanded_multipolygon.append([Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])])
            else:
                expanded_polygon = []
                expanded_all_polygon = []
                for j in range(1, int(TTC * sampling_fq)):
                    new_theta_ = yaw_[i] + j / sampling_fq * yaw_rate[i]
                    if yaw_rate[i] != 0:
                        radius = np.sqrt(vx_[i] ** 2 + vy_[i] ** 2) / yaw_rate[i]
                        new_x_ = x_[i] + radius * ( np.sin(new_theta_) - np.sin(yaw_[i]) )
                        new_y_ = y_[i] + radius * ( - np.cos(new_theta_) + np.cos(yaw_[i]) )
                    else:
                        # for the case of yaw_rate = 0, constant velocity model is used
                        new_x_ = x_[i] + j / sampling_fq * vx_[i]
                        new_y_ = y_[i] + j / sampling_fq * vy_[i]
                    # project new_theta_ to (-pi,pi)
                    new_theta_ = self.__project_angel(new_theta_)
                    expanded_polygon.append(self.instant_polygon(new_x_, new_y_, new_theta_, length_[i], width_[i]))
                    expanded_all_polygon.append(expanded_polygon[j - 1])
                # expanded_polygon_set.append(MultiPolygon(expanded_polygon))
                expanded_polygon_set.append(unary_union(expanded_polygon).buffer(0.01))
                self.expanded_multipolygon.append(expanded_all_polygon)
        return expanded_polygon_set

    def expanded_bbox_list(self, expand: float = 2.0):
        expanded_bbox_list = []
        x_ = self.kinematics['x']
        y_ = self.kinematics['y']
        yaw_ = self.kinematics['bbox_yaw']
        length_ = self.kinematics['length']
        width_ = self.kinematics['width']
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                expanded_bbox_list.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
                self.expanded_bboxes.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))
            else:
                expanded_bbox = self.instant_polygon(x_[i], y_[i], yaw_[i], expand * length_[i], expand * width_[i])
                expanded_bbox_list.append(expanded_bbox)
                self.expanded_bboxes.append(expanded_bbox)
        return expanded_bbox_list
